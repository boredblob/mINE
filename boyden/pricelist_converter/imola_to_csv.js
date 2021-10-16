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
  
  // Output pricelist as CSV
  const csv = new ObjectsToCsv(pricelist);
  await csv.toDisk('./imola.csv');
  //console.log(await csv.toString());
});

function productStringToObject(str) {
  const productLine = str.split(" ");
  // find index of size string
  const sizeIndex = productLine.findIndex(e => {
    return e.includes("x");
  });

  if (productLine.length-sizeIndex !== 10 && productLine.length-sizeIndex !== 11) {
    console.log("Error: Product lines not equal lengths.");
    console.log(productLine);
    return "ERROR";
  }

  if (productLine.length-sizeIndex !== 11) {// offset table array indices after Um. by one if 2^ Sc price missing
    productLine.splice(sizeIndex+3, 0, "");
  }

  const productDetails = productLine.slice(sizeIndex);

  // swap commas and full stops
  for (let i = 2; i < productDetails.length; i++) {
    productDetails[i] = productDetails[i]
      .replaceAll(".", "")
      .replaceAll(",", ".")
  }

  const size = productDetails[0].split("x").map(x => x*10);
  const tilesPerSqm = 1000000 / (size[0] * size[1]);

  // TODO: Change formulas based on PZ/MQ
  if (productDetails[1] !== "PZ" && productDetails[1] !== "MQ") {
    console.log("Error: Um not recognised");
    console.log(productLine)
    return "ERROR";
  }

  const priceIsPerSqm = (productDetails[1] === "MQ"); // Change price calc based on initial price units

  const netPricePerTile = priceIsPerSqm ? round(productDetails[2] / tilesPerSqm, 2): parseFloat(productDetails[2]);
  const netPricePerSqm = priceIsPerSqm ? parseFloat(productDetails[2]) : round(productDetails[2] * tilesPerSqm, 2);

  const product = {
    code: productLine.slice(0, sizeIndex).join(" "),
    size: productDetails[0],
    netPricePerSqm: netPricePerSqm,
    netPricePerTile: netPricePerTile,
    costPricePerTile: round(netPricePerTile * 0.35, 2),
    boxQuantity: parseInt(productDetails[4]),
    tileWeight: ceil(productDetails[6] / productDetails[4], 2),
    tilesPerPallet: round(productDetails[7] * productDetails[4], 2),
    width: size[0],
    height: size[1]
  };

  return product;
}

function round(num, dp) {
  return Math.round((num + Number.EPSILON) * (10**dp)) / (10**dp);
}

function ceil(num, dp) {
  return Math.ceil(num*(10**dp)) / (10**dp);
}