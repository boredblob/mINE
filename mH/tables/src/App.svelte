<script>
  let rows = 2;
  let cols = 3;

  let table = [];
  $: {
    let oldRows = table.length || 0;
    let oldCols = table[0]?.length || 0;
    if (oldRows !== rows || oldCols !== cols) {
      if (rows >= oldRows && cols >= oldCols) {
        for (let r = oldRows; r < rows; r++) {
          table.push(Array(cols).fill(""));
        }
        for (let c = oldCols; c < cols; c++) {
          for (let r of table) {
            if (r.length < cols) r.push("");
          }
        }
      } else {
        for (let i = 0; i < oldRows - rows; i++) {
          table.pop();
        }
        for (let i = 0; i < oldCols - cols; i++) {
          for (let r of table) {
            r.pop();
          }
        }
      }
      table = table;
    }
  }

  function calculateOutput(tab) {
    let str = "[table]";
    for (let row of tab) {
      str += "[r]";
      for (let item of row) {
        let itemStr = (item == "") ? " " : item;
        str += "[c]" + itemStr + "[/c]";
      }
      str += "[/r]";
    }
    str += "[/table]";
    return str;
  }
</script>

<main>
  <header>
    <input type="number" name="rows" id="rows" placeholder="Rows" value={rows} on:input={e => {if (e.target.value > 0) rows = Math.floor(Number(e.target.value))}}>
    <input type="number" name="columns" id="columns" placeholder="Columns" value={cols} on:input={e => {if (e.target.value > 0) cols = Math.floor(Number(e.target.value))}}>
  </header>
  <table>
    {#each table as row, r}
      <tr>
        {#each row as item, c}
          <td>
            <textarea name={`t-${r*rows+c}`} id={`t-${r*rows+c}`} data-row={r} data-col={c} on:input={e => table[r][c] = e.target.value}></textarea>
          </td>
        {/each}
      </tr>
    {/each}
  </table>
  <footer>
    <p class="output">{calculateOutput(table)}</p>
    <button class="copy" on:click={e => navigator.clipboard.writeText(calculateOutput(table)).then(function() {
      e.target.innerHTML = "Copied!";
      setTimeout(() => e.target.innerHTML = "Copy Code", 750);
    })}>Copy Code</button>
  </footer>
</main>

<style>
  * {
    border: none;
    padding: none;
  }

  main {
    display: grid;
    grid-template-columns: 1fr;
    grid-template-rows: min-content auto min-content;
    height: 100%;
    width: 100%;
  }

	header, footer {
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 1rem 0;
  }

  header {
    border-bottom: 1px solid #ccc;
  }

  footer {
    border-top: 1px solid #ccc;
    flex-direction: column;
  }

  header>input {
    width: 4rem;
    text-align: center;
    margin: 0 1rem;
  }

  table {
    width: 100%;
    padding: 1rem 2rem;
    height: 100%;
    box-sizing: border-box;
  }

  table>tr>td>textarea {
    height: 100%;
    width: 100%;
    padding: 0;
    margin: 0;
    border-radius: 0;
  }

  footer>p {
    width: 100%;
    text-align: center;
  }

  button.copy {
    width: 6rem;
  }
</style>