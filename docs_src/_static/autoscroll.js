window.addEventListener('DOMContentLoaded', () => {

  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      const id = entry.target.getAttribute('id');
      // console.log('Scrolled to')
      // console.log(entry.target)
      if (entry.intersectionRatio > 0) {
        document.querySelector(`.localtoc li a[href="#${id}"]`).parentElement.classList.add('active');
      } else {
        document.querySelector(`.localtoc li a[href="#${id}"]`).parentElement.classList.remove('active');
      }
    });
  });

  // Track all sections that have an `id` applied
  document.querySelectorAll('dt[id]').forEach((section) => {
    observer.observe(section);
  });
  document.querySelectorAll('section[id]').forEach((section) => {
    observer.observe(section);
  });
  
});