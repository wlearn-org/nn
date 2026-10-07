# Contributing to @wlearn/nn

## Tests

```sh
npm install
npm test
```

`npm run test:migration` checks saved models across JavaScript and Python, in
both directions, together with pipelines, batching and resource cleanup. It
needs:

- `WLEARN_PYTHON`: a Python executable that can import `polygrad` and `wlearn`.
- `POLY_LIB`: path to a locally built `libpolygrad.so`, when testing a local
  Polygrad build.
- `WLEARN_NN_TEST_CORE=wasm` (optional) to test the WebAssembly core instead of
  the native one.

The test writes its temporary files into a fresh directory for each run.
