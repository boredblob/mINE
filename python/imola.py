from numpy import true_divide
import tabula
import pandas as pd

pdf = "./pricelist.pdf"

ranges = tabula.read_pdf(pdf, pages="4")[0]["progetto-projects"].tolist()

tables = tabula.read_pdf(pdf, pages="5-86")
table = pd.concat(tables, join="inner")

def isInRange(value):
  for i in ranges:
    if (i in str(value)): return True
  return False

# remove lines without data
table = table.loc[[(not str(item).startswith("..")) for item in table["Unnamed: 0"]]]
table = table.loc[(table['MQ'].str.len() > 0) | (table['PALLET / CRATE *'].str.len() > 0)]
table = table.loc[table["Unnamed: 0"] != "Codice"]

table.to_csv("c.csv")

pricelist = []

# format into: code, size, uom, sqm price, box qty, box weight, boxes/pallet, pallet weight
for index, r in table.iterrows():
  row = r.tolist()
  product = [row[0]]
  if (str(row[4]) == "nan"):
    product.extend((row[1], row[2], row[3]))
    if (str(row[5]) == "nan"):
      product.extend((
        row[7].split()[0], # box qty
        row[6], # box weight
        row[9].split()[0], # boxes/pallet
        row[9].split()[-1] # pallet weight
      ))
    else:
      if ("," in str(row[5])):
        product.extend((
          row[6], # box qty
          row[8], # box weight
          row[9].split()[0], # boxes/pallet
          row[9].split()[-1] # pallet weight
        ))
      else:
        product.extend((
          row[5], # box qty
          row[6], # box weight
          row[9].split()[0], # boxes/pallet
          row[9].split()[-1] # pallet weight
        ))
  else:
    product.extend((row[4], row[7], row[9], row[1], row[3], row[5], row[8]))

  # convert numbers between italian decimal notation and english
  for i in range(3, len(product)):
    product[i] = product[i].replace(".", "").replace(",", ".").replace("* ", "")

  width = product[1].split("x")[0]
  height = product[1].split("x")[1]
  width = float(width) * 10
  height = float(height) * 10
  product.append(width)
  product.append(height)

  # add price per tile
  if (product[2] == "MQ"):
    product.insert(4, round(float(product[3]) / (1000000/width/height), 2))
  else:
    product.insert(4, round(float(product[3]), 2))

  pricelist.append(product)

fields = ["Product Code", "Size", "UOM", "Net Price per UOM", "Net Price per Tile", "Box Qty", "Box Weight", "Boxes per Pallet", "Pallet Weight", "Width", "Height"]

pd.DataFrame(pricelist, columns=fields).to_csv("b.csv")

# table = table.loc[[isInRange(item) for item in table["Unnamed: 0"]]]
# table.to_csv("a.csv")