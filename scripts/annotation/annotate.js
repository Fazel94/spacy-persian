// LLM NER annotator. Runs inside the harness JS eval kernel because `completion()` only
// exists there. Invoke from an eval cell:
//
//   globalThis.ANNOTATE = {
//     jobs: [{ requests: "<abs>/…/shard-000.jsonl", responses: "<abs>/…/shard-000.jsonl" }, …],
//     cache: "<abs>/annotation/work/cache",
//     model: "default", concurrency: 16,
//   };
//   await (0, eval)(await Bun.file("<abs>/scripts/annotation/annotate.js").text());
//
// A single `{requests, responses}` pair may be given at the top level instead of `jobs`.
// Every batch of every job goes into ONE work queue, so `concurrency` is the real number of
// in-flight model calls no matter how the work is split across files; shard boundaries cost
// nothing in throughput.
//
// Indirect eval runs in global scope, so `completion` resolves. Everything lives inside one
// async IIFE so the file can be evaluated repeatedly without lexical redeclaration errors.
//
// Per request line: cache hit -> reuse; else call the model, verify the returned id set
// equals the requested one, retry once on mismatch, then split the batch in half and
// re-issue each half. A call that throws (rate limit, transport) is retried with backoff and
// then recorded as failed; failures are never cached, so re-running resumes exactly the
// batches that did not land.

(async () => {
  const cfg = globalThis.ANNOTATE;
  if (!cfg) throw new Error("set globalThis.ANNOTATE before evaluating annotate.js");
  if (typeof completion === "undefined") {
    throw new Error("completion() is not visible; paste this file's body into the eval cell");
  }
  const model = cfg.model ?? "default";
  const concurrency = cfg.concurrency ?? 8;
  const attempts = cfg.attempts ?? 3;
  const jobs = cfg.jobs ?? [{ requests: cfg.requests, responses: cfg.responses }];
  const INSTRUCTION = "Annotate every sentence. Return one item per id.";

  const { mkdir, writeFile } = await import("node:fs/promises");
  const path = await import("node:path");

  await mkdir(cfg.cache, { recursive: true });
  const queue = [];
  for (const [j, job] of jobs.entries()) {
    const text = await Bun.file(job.requests).text();
    const reqs = text.split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
    job.results = new Array(reqs.length);
    await mkdir(path.dirname(job.responses), { recursive: true });
    reqs.forEach((req, i) => queue.push({ job: j, index: i, req }));
  }

  const stats = { cached: 0, called: 0, retried: 0, errored: 0, failed: 0 };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const parse = (raw) => (typeof raw === "string" ? JSON.parse(raw) : raw);

  const bodyFor = (req, ids) => {
    const byId = new Map(req.sentences.map((s) => [s.id, s.text]));
    return INSTRUCTION + "\n\n" + ids.map((id) => `${id}: ${byId.get(id)}`).join("\n");
  };

  // One model call, retried through transport/rate-limit errors before giving up.
  const ask = async (req, ids) => {
    for (let a = 1; ; a++) {
      try {
        stats.called += 1;
        const out = parse(
          await completion(bodyFor(req, ids), { model, system: req.system, schema: req.schema }).wait(),
        );
        const items = Array.isArray(out?.items) ? out.items : [];
        const want = new Set(ids);
        const got = new Set(items.map((i) => i.id));
        const ok = got.size === want.size && [...want].every((id) => got.has(id));
        return { items: items.filter((i) => want.has(i.id)), ok };
      } catch (err) {
        stats.errored += 1;
        if (a >= attempts) {
          console.log(`  call failed on ${req.batch} after ${a} attempts: ${err?.message ?? err}`);
          return { items: [], ok: false, threw: true };
        }
        await sleep(1000 * 2 ** (a - 1));
      }
    }
  };

  const run = async (req, ids, depth = 0) => {
    let attempt = await ask(req, ids);
    if (attempt.ok) return { items: attempt.items, error: null };
    // A thrown call (transport, rate limit, content refusal) is not necessarily the whole
    // batch's fault: a single sentence can trigger a refusal. Bisect so the rest still lands.
    if (attempt.threw && ids.length === 1) return { items: [], error: "call-failed" };
    if (!attempt.threw && depth === 0) {
      stats.retried += 1;
      attempt = await ask(req, ids);
      if (attempt.ok) return { items: attempt.items, error: null };
    }
    if (ids.length > 1 && (attempt.threw || depth < 2)) {
      const mid = Math.ceil(ids.length / 2);
      const [ra, rb] = [
        await run(req, ids.slice(0, mid), depth + 1),
        await run(req, ids.slice(mid), depth + 1),
      ];
      const errs = [ra.error, rb.error].filter(Boolean);
      return { items: [...ra.items, ...rb.items], error: errs.length ? errs[0] : null };
    }
    console.log(`  id-mismatch on ${req.batch} ids=${ids.join(",")}`);
    return { items: attempt.items, error: "id-mismatch" };
  };

  let next = 0;
  const worker = async () => {
    for (;;) {
      const slot = next++;
      if (slot >= queue.length) return;
      const { job: j, index, req } = queue[slot];
      const cacheFile = path.join(cfg.cache, `${req.cache_key}.json`);
      if (await Bun.file(cacheFile).exists()) {
        jobs[j].results[index] = JSON.parse(await Bun.file(cacheFile).text());
        stats.cached += 1;
        continue;
      }
      if (cfg.cacheOnly) {
        jobs[j].results[index] = {
          batch: req.batch, model, prompt_hash: req.prompt_hash, items: [],
          error: "uncached", failed_ids: req.ids,
        };
        stats.failed += 1;
        continue;
      }
      const { items, error } = await run(req, req.ids);
      const record = {
        batch: req.batch,
        model,
        prompt_hash: req.prompt_hash,
        items,
        error,
        ...(error ? { failed_ids: req.ids.filter((id) => !items.some((it) => it.id === id)) } : {}),
      };
      jobs[j].results[index] = record;
      if (error) stats.failed += 1;
      else await writeFile(cacheFile, JSON.stringify(record), "utf8");
    }
  };

  const t0 = Date.now();
  await Promise.all(Array.from({ length: concurrency }, worker));

  for (const job of jobs) {
    await writeFile(job.responses, job.results.map((r) => JSON.stringify(r)).join("\n") + "\n", "utf8");
  }

  const secs = (Date.now() - t0) / 1000;
  const line = `jobs=${jobs.length} batches=${queue.length} cached=${stats.cached} called=${stats.called}`
    + ` retried=${stats.retried} call_errors=${stats.errored} failed=${stats.failed}`
    + ` concurrency=${concurrency} elapsed=${secs.toFixed(1)}s (${(queue.length / secs).toFixed(2)} batches/s)`;
  console.log(line);
  return line;
})()
