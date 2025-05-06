const select = document.getElementById('position-select');
const container = document.getElementById('table-container');

let data = {};

fetch('ppg_data.json')
  .then(res => res.json())
  .then(json => {
    data = json;

    select.addEventListener('change', () => {
      const pos = select.value;
      container.innerHTML = '';
      if (!pos || !data[pos]) return;

      const tbl = document.createElement('table');
      const header = tbl.insertRow();
      ['Rank', 'Player', 'PPG'].forEach(h => {
        const th = document.createElement('th');
        th.textContent = h;
        header.appendChild(th);
      });

      data[pos].forEach(row => {
        const tr = tbl.insertRow();
        tr.insertCell().textContent = row.rank;
        tr.insertCell().textContent = row.player;
        tr.insertCell().textContent = row.ppg;
      });

      container.appendChild(tbl);
    });
  })
  .catch(err => {
    console.error('Failed to load PPG data:', err);
    container.textContent = 'Error loading data.';
  });
