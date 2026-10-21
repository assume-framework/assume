// Publishes the height of the sticky global controls as --controls-height so
// that other sticky elements can dock directly underneath them.
(function () {
  function observe() {
    var controls = document.querySelector(".controls");
    if (!controls) {
      setTimeout(observe, 200);
      return;
    }
    var update = function () {
      document.documentElement.style.setProperty(
        "--controls-height",
        controls.getBoundingClientRect().height + "px"
      );
    };
    new ResizeObserver(update).observe(controls);
    update();
  }
  observe();
})();
