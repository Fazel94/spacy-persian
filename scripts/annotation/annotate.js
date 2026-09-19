// LLM NER annotator. Runs inside the harness JS eval kernel because `completion()` only
// exists there. Invoke from an eval cell:
//
//   globalThis.ANNOTATE = {
//     requests: "<abs>/annotation/work/requests/select-100.jsonl",
//     responses: "<abs>/annotation/work/responses/select-100.jsonl",
//     cache: "<abs>/annotation/work/cache",
//     model: "default", concurrency: 4,
//   };
//   await (0, eval)(await Bun.file("<abs>/scripts/annotation/annotate.js").text());
//
// Indirect eval runs in global scope, so `completion` resolves. Everything lives inside one
// async IIFE so the file can be evaluated repeatedly without lexical redeclaration errors.
//
// Per request line: cache hit -> reuse; else call the model, verify the returned id set
// equals the requested one, retry once on mismatch, then split the batch in half and
// re-issue each half. A half that still mismatches is recorded with error "id-mismatch".

(async () => {
  const cfg = globalThis.ANNOTATE;
  if (!cfg) throw new Error("set globalThis.ANNOTATE before evaluating annotate.js");
  if (typeof completion === "undefined") {
    throw new Error("completion() is not visible; paste this file's body into the eval cell");
  }
  const model = cfg.model ?? "default";
  const concurrency = cfg.concurrency ?? 4;
  const INSTRUCTION = "Annotate every sentence. Return one item per id.";

  const { mkdir, writeFile } = await import("node:fs/promises");
  const path = await import("node:path");

  const text = await Bun.file(cfg.requests).text();
  const requests = text.split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
  await mkdir(cfg.cache, { recursive: true });
  await mkdir(path.dirname(cfg.responses), { recursive: true });

  const stats = { cached: 0, called: 0, retried: 0, failed: 0 };

  const parse = (raw) => (typeof raw === "string" ? JSON.parse(raw) : raw);

  const bodyFor = (req, ids) => {
    const byId = new Map(req.sentences.map((s) => [s.id, s.text]));
    return INSTRUCTION + "\n\n" + ids.map((id) => `${id}: ${byId.get(id)}`).join("\n");
  };

  const ask = async (req, ids) => {
    stats.called += 1;
    const out = parse(
      await completion(bodyFor(req, ids), { model, system: req.system, schema: req.schema }).wait(),
    );
    const items = Array.isArray(out?.items) ? out.items : [];
    const got = new Set(items.map((i) => i.id));
    const want = new Set(ids);
    const ok = got.size === want.size && [...want].every((id) => got.has(id));
    return { items: items.filter((i) => want.has(i.id)), ok };
  };

  // Returns {items, error}. Splits recursively-once: whole batch, retry, then halves.
  const run = async (req, ids, depth = 0) => {
    let attempt = await ask(req, ids);
    if (attempt.ok) return { items: attempt.items, error: null };
    if (depth === 0) {
      stats.retried += 1;
      attempt = await ask(req, ids);
      if (attempt.ok) return { items: attempt.items, error: null };
    }
    if (ids.length > 1 && depth < 2) {
      const mid = Math.ceil(ids.length / 2);
      const [a, b] = [ids.slice(0, mid), ids.slice(mid)];
      const [ra, rb] = [await run(req, a, depth + 1), await run(req, b, depth + 1)];
      const errs = [ra.error, rb.error].filter(Boolean);
      return { items: [...ra.items, ...rb.items], error: errs.length ? errs[0] : null };
    }
    console.log(`  id-mismatch on ${req.batch} ids=${ids.join(",")}`);
    return { items: attempt.items, error: "id-mismatch" };
  };

  const results = new Array(requests.length);
  let next = 0;

  const worker = async () => {
    for (;;) {
      const i = next++;
      if (i >= requests.length) return;
      const req = requests[i];
      const cacheFile = path.join(cfg.cache, `${req.cache_key}.json`);
      if (await Bun.file(cacheFile).exists()) {
        results[i] = JSON.parse(await Bun.file(cacheFile).text());
        stats.cached += 1;
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
      results[i] = record;
      if (error) stats.failed += 1;
      else await writeFile(cacheFile, JSON.stringify(record), "utf8");
      console.log(`  ${req.batch}: ${items.length}/${req.ids.length} ids${error ? " " + error : ""}`);
    }
  };

  await Promise.all(Array.from({ length: concurrency }, worker));

  await writeFile(
    cfg.responses,
    results.map((r) => JSON.stringify(r)).join("\n") + "\n",
    "utf8",
  );

  const line = `batches=${requests.length} cached=${stats.cached} called=${stats.called} retried=${stats.retried} failed=${stats.failed}`;
  console.log(line);
  return line;
})()
