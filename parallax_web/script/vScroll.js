window.addEventListener('scroll', function(event) {
    var topDistance = window.pageYOffset;
    var layers = document.querySelectorAll("[data-type='parallax']");
  
    layers.forEach(function(layer) {
      var depth = layer.getAttribute('data-depth');
      var movement = -(topDistance * depth);
      var translate3d = 'translate3d(0, ' + movement + 'px, 0)';
      layer.style.transform = translate3d;
    });
  });
  