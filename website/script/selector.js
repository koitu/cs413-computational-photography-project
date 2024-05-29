document.addEventListener('DOMContentLoaded', function() {
    const selector = document.getElementById('imageSelector');
    selector.addEventListener('change', function() {
      const value = this.value;
      // Logging the value to check if the switch case is entered correctly
      console.log("Selected value:", value);
      switch (value) {
        case 'lake':
          document.querySelector('.layer-bg').style.backgroundImage = 'url("../images/lake_2.png")';
          document.querySelector('.layer-1').style.backgroundImage = 'url("../images/lake_1.png")';
          document.querySelector('.layer-2').style.backgroundImage = 'url("../images/lake_0.png")';
          break;
        case 'gardenCat':
          document.querySelector('.layer-bg').style.backgroundImage = 'url("../images/gardenCat_2.png")';
          document.querySelector('.layer-1').style.backgroundImage = 'url("../images/gardenCat_1.png")';
          document.querySelector('.layer-2').style.backgroundImage = 'url("../images/gardenCat_0.png")';
          break;
        case 'snowmountain':
          document.querySelector('.layer-bg').style.backgroundImage = 'url("../images/snowmountain_2.png")';
          document.querySelector('.layer-1').style.backgroundImage = 'url("../images/snowmountain_1.png")';
          document.querySelector('.layer-2').style.backgroundImage = 'url("../images/snowmountain_0.png")';
          break;
      }
    });
  });
  