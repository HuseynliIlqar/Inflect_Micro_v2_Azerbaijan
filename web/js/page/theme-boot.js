// The saved theme on <html>, before the first paint, so a dark choice never
// flashes white. A classic script loaded without `defer` from <head>, not a
// module: a module runs after the first paint, which is the flash this file
// exists to prevent. Every page links it -- index.html and the About pages.
// The same rule as initialTheme() in js/ui/theme.js.
(function () {
  var theme = null;
  try {
    theme = localStorage.getItem("theme");
  } catch (e) {}
  if (theme !== "light" && theme !== "dark") {
    theme = new URLSearchParams(location.search).get("__theme");
  }
  if (theme === "light" || theme === "dark") {
    document.documentElement.setAttribute("data-theme", theme);
  }
})();
