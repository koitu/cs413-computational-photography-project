document.addEventListener("DOMContentLoaded", function() {
    const layers = document.querySelectorAll('.layer[data-type="parallax"]');
    document.addEventListener("mousemove", function(e) {
        const _w = window.innerWidth / 2;
        const _h = window.innerHeight / 2;
        const _mouseX = e.clientX;
        const _mouseY = e.clientY;

        layers.forEach(layer => {
            const depth = layer.getAttribute('data-depth');
            const movementX = (_mouseX - _w) * depth * 1.2;
            const movementY = (_mouseY - _h) * depth * 1.2;
            const translate3d = `translate3d(${movementX}px, ${movementY}px, 0px)`;
            layer.style.transform = translate3d;
        });
    });
});
