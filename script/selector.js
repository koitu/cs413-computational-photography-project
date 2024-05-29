document.addEventListener('DOMContentLoaded', function() {
  const selector = document.getElementById('imageSelector');
  const lakeDiv = document.getElementById('tunnelBook-lake');
  const gardenCatDiv = document.getElementById('tunnelBook-gardenCat');
  const snowMountainDiv = document.getElementById('tunnelBook-snowmountain');

  function updateVisibility() {
      lakeDiv.classList.add('hidden');
      gardenCatDiv.classList.add('hidden');
      snowMountainDiv.classList.add('hidden');

      const value = selector.value;
      switch (value) {
          case 'lake':
              lakeDiv.classList.remove('hidden');
              break;
          case 'gardenCat':
              gardenCatDiv.classList.remove('hidden');
              break;
          case 'snowmountain':
              snowMountainDiv.classList.remove('hidden');
              break;
      }
  }

  selector.addEventListener('change', updateVisibility);
  updateVisibility(); 
});
