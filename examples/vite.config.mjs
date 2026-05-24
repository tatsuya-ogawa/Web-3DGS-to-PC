export default {
  base: "./",
  build: {
    rollupOptions: {
      input: {
        index: new URL("./index.html", import.meta.url).pathname,
        convert: new URL("./convert.html", import.meta.url).pathname,
        tps: new URL("./tps.html", import.meta.url).pathname,
      },
    },
  },
};
