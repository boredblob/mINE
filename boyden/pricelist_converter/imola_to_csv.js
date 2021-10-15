const ObjectsToCsv = require('objects-to-csv');
const fs = require('fs');

fs.readFile("./pricelists/imola.txt", "utf-8", async (err, data) => {
  if (err) return console.log(err);
  let list = data
    .replace(/(.*[a-wyz].*)[\n\r]*/gm, "")
    .replace(/(.*GBP.*)[\n\r]*/gm, "")
    .replace(/(.*\.\..*)[\n\r]*/gm, "")
    .replace(/^(?!.*x.*).+$[\n\r]*/gm, "")
    .replace(/(\*\s)/g, "")
    .split("\n");
  const pricelist = [];
  for (let product of list) { //list.splice(0, 10)
    pricelist.push(productStringToObject(product));
  }
  // for (let i = 0; i < 10; i++) {
  //   console.log(pricelist[i]);
  // }
  // for (let i = list.length-10; i < list.length; i++) {
  //   console.log(pricelist[i]);
  // }
  
  // Output pricelist as CSV
  const csv = new ObjectsToCsv(pricelist);
  await csv.toDisk('./imola.csv');
  console.log(await csv.toString());
});

function productStringToObject(str) {
  const productLine = str.replaceAll(",", ".").split(" ");
  // find index of size string
  const sizeIndex = productLine.findIndex(e => {
    return e.includes("x");
  });

  const size = productLine[sizeIndex].split("x");
  const tilesPerSqm = 10000 / (size[0] * size[1]);

  const netPricePerTile = round(productLine[sizeIndex+2] / tilesPerSqm, 2);

  return {
    code: productLine.slice(0, sizeIndex).join(" "),
    size: productLine[sizeIndex+0],
    netPricePerSqm: parseFloat(productLine[sizeIndex+2]),
    netPricePerTile: netPricePerTile,
    costPricePerTile: round(netPricePerTile * 0.35, 2),
    boxQuantity: parseInt(productLine[sizeIndex+4]),
    tileWeight: round(productLine[sizeIndex+6] / productLine[sizeIndex+4], 2),
    tilesPerPallet: round(productLine[sizeIndex+7] * productLine[sizeIndex+4], 2)
  }
}

function round(num, dp) {
  return Math.round((num + Number.EPSILON) * (10**dp)) / (10**dp);
}