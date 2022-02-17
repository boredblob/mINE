import pandas
import re
import xlsxwriter

table = pandas.io.parsers.read_csv('stock products.csv')
displays = open('displays.txt', 'r').readline().split(',')
ranges = {}
sheets = {}
info = {}
for d in displays:
  ranges[d] = []
  sheets[d] = []

# make filter array to include only those matching displays
filter_arr = []
for _, row in table.iterrows():
  found = False
  for d in displays:
    contains = (d + " ") in row['Description'].lower()
    if contains:
      found = True
      break
  filter_arr.append(found)

for _, row in table.loc[filter_arr].iterrows():
  description = row['Description'].lower()
  display = ""
  for d in displays:
    if (d + " ") in description:
      display = d
      break


  match = re.search('[0-9]+x[0-9]+(x[0-9\.]+)?', description)
  filter_arr.append(match != None)
  if (match != None):
    size = match.group().split('x')
    width = str(min(int(size[0]), int(size[1])))
    height = str(max(int(size[0]), int(size[1])))
    filtered_colour = description[re.search(display, description).span()[1]:match.span()[0]]
    match2 = re.search('([a-z-][a-z-][a-z-]+ ?)+', filtered_colour)
    if (match2 != None):
      filtered_colour = re.sub('((lapato)|(channel)|(floor)|(wall)|(rectified)|(metal)|(edison)|(large)|(sense)|(land)|(absolute)|(textures)|(split)|(relieve)|(reed)) *', '', match2.group()).strip()
      list_colour = re.sub('((decor)|(semi-polished)|(polished)|(matt)|(satin)|(gloss)|(bevel)|(brick)|(anti-slip)|(bumpy)|(hex)|(natural)) *', '', filtered_colour).strip()
      ranges[display].append([row, filtered_colour, width, height, list_colour])

# remove non-stocked items from stocked ranges
for r in ranges:
  numStocked = 0
  for p in ranges[r]:
    if (p[0]['Stock Rank'] == 'Primary Stock Ranking') | (p[0]['Stock Rank'] == 'Secondary Stock Ranking'):
      numStocked = numStocked + 1
  if (numStocked > 0):
    if (numStocked != len(ranges[r])):
      ranges[r] = [p for p in ranges[r] if (p[0]['Stock Rank'] == 'Primary Stock Ranking') | (p[0]['Stock Rank'] == 'Secondary Stock Ranking')]

def appendIfNotContained(arr, index, priority):
  for x in arr:
    if (x[0] == index):
      x[1] = max(x[1], priority)
      return
  arr.append([index, priority])

def getAllDescriptions(arr):
  arr2 = []
  for p in arr:
    arr2.append(p[1] + " " + p[2] + " x " + p[3])
  return arr2

def splitSubArray(arr, subarr, indexOfSubArr, splitIndex):
  arr.insert(indexOfSubArr+1, subarr[splitIndex:])
  arr.insert(indexOfSubArr+1, subarr[:splitIndex])
  arr.pop(indexOfSubArr)

# generate possible places to split arrays
for r in ranges:
  sorted_range = sorted(ranges[r], key=lambda p: ((p[0]['Material'] == 'Ceramic'), not ('decor' in p[1]), float(p[2])*float(p[3]), p[4], p[1]))
  splits = []

  current_material = sorted_range[0][0]['Material']
  current_size = sorted_range[0][2]+"x"+sorted_range[0][3]
  current_decor = ('decor' in sorted_range[0][1])

  for i, p in enumerate(sorted_range):
    if (p[0]['Material'] != current_material):
      appendIfNotContained(splits, i, 3)
    if (('decor' in p[1]) != current_decor):
      appendIfNotContained(splits, i, 2)
    if (p[2]+"x"+p[3] != current_size):
      appendIfNotContained(splits, i, 1)
    current_material = p[0]['Material']
    current_decor = ('decor' in p[1])
    current_size = p[2]+"x"+p[3]

  # split tiles in range into sheets of 4 smartly where possible
  split_range = [sorted_range]
  for split in sorted(splits, reverse=True, key=lambda x: (x[1])):
    index = 0
    for i, _range in enumerate(split_range):
      if (i>0): index = index + len(split_range[i-1])
      if (len(_range) > 4):
        if ((split[0] >= index) & (split[0] < index+len(_range))):
          splitSubArray(split_range, _range, i, split[0]-index)
          break

  # split tiles in range into sheets of 4 less smartly
  for i, sheet in enumerate(split_range):
    if (len(sheet) > 4):
      # offset first item if it is unique (did this because the room black doesn't have a matt version)
      firstMatches = 0
      secondMatches = 0
      for p in sheet:
        if (sheet[0][1].split()[0] in p[1]):
          firstMatches = firstMatches + 1
      for p in sheet:
        if (sheet[1][1].split()[0] in p[1]):
          secondMatches = secondMatches + 1
      if ((firstMatches < 2) & (secondMatches > 1)):
        split_range[i].append(split_range[i].pop(0))

      splitSubArray(split_range, sheet, i, 4)

  # merge sheets that are less than 4 long if they add up okay
  for i, sheet in enumerate(split_range):
    for j, sheet2 in enumerate(split_range):
      if (i != j):
        if (len(sheet) + len(sheet2) <= 4):
          sheet.extend(sheet2)
          split_range.pop(j)

  sheets[r] = [*split_range]


filenames_set = set([]) # ensure no duplicate filenames

for d in displays:
  colours_set = set([])
  sizes_set = set([])
  finishes_set = set([])
  decor = any([('decor' in p[1]) for p in ranges[d]])
  for p in ranges[d]:
    colours_set.add(p[4].strip().title())
    sizes_set.add((p[2],p[3]))
    finishes_set.add(p[0]['Finish'].split('/')[0].strip().title())
  info[d] = {
    'colours': sorted(list(colours_set), key=lambda c: (c=="Hop Mix", c)),
    'sizes': [x[0] + "x" + x[1] for x in sorted(list(sizes_set), key=lambda y: (int(y[0])*int(y[1])))],
    'finishes': sorted(list(finishes_set))
  }

  filenames_list = []
  for i, sheet in enumerate(sheets[d]):
    sheet_sizes = set([])
    for p in sheet:
      sheet_sizes.add(p[2]+"x"+p[3])


    # if current sheet is just one size
    # and more than one sheet is just this size
    # add i at the end of the filename
    if (len(list(sheet_sizes)) == 1):
      if (len(info[d]['sizes']) == 1):
        if len(sheets[d]) > 1:
          filename = d.title()
        else:
          filename = d.title()
      else:
        if (decor and all([('decor' in p1[1]) for p1 in sheet])):
          filename = d.title() + " Decor"
        else:
          if (sum([(all([list(sheet_sizes)[0] == p[2]+"x"+p[3] for p in other_sheet]) and not all(['decor' in p[1] for p in other_sheet])) for other_sheet in sheets[d]]) > 1):
            filename = d.title() + " " + list(sheet_sizes)[0]
          else:
            filename = d.title() + " " + list(sheet_sizes)[0]
    else:
      if len(sheets[d]) > 1:
        filename = d.title()
      else:
        filename = d.title()

    filenames_list.append(filename)

  # add numbers to filename if sizes are duplicated
  for size in info[d]['sizes']:
    instances_of_size = []
    for i, filename in enumerate(filenames_list):
      if (size in filename):
        instances_of_size.append(i)
    if (len(instances_of_size) > 1):
      for i, instance in enumerate(instances_of_size):
        # print("Duplicate sizes: ", filenames_list[instance])
        filenames_list[instance] += (" " + str(i+1))

  # then add new numbers to the filenames if there are still duplicates
  instances_of_name = set([])
  for i, filename1 in enumerate(filenames_list):
    for j, filename2 in enumerate(filenames_list):
      if (i != j):
        if (filename1 == filename2):
          # print("Duplicate names: ", filename1, filename2)
          instances_of_name.add(i)
          instances_of_name.add(j)
  if (len(instances_of_name) > 1):
    for i, instance in enumerate(list(instances_of_name)):
      filenames_list[instance] += (" " + str(i+1))
  
  # warn about duplicate filenames
  for filename in filenames_list:
    if (filename in filenames_set): print('WARNING: Duplicate Filename:', filename) # warn about duplicates
    filenames_set.add(filename)

  for i, sheet in enumerate(sheets[d]):
    workbook = xlsxwriter.Workbook('./generated/' + filenames_list[i] + '.xlsx')
    worksheet = workbook.add_worksheet('Sheet1')
    worksheet.set_margins(.25, .25, .75, .75)

    wrap = {'text_wrap': True}
    border = {'border': 1}
    center = {'align': 'center', 'valign': 'vcenter'}
    bold = {'bold': True}

    def tick_cell(row, col, y_offset=0):
      tick_scale = 1/3
      tick_format = {'x_scale': tick_scale, 'y_scale': tick_scale, 'x_offset': 10, 'y_offset': -6 + y_offset}
      worksheet.insert_image(row, col, 'tick.png', tick_format)

    fTitle = workbook.add_format({**wrap, **center, **border, **bold, 'font_name': 'Aparajita', 'font_size': 26, 'font_color': 'white', 'bg_color': 'black'})

    fBorder = workbook.add_format({**wrap, **border})

    fCenter6 = workbook.add_format({**wrap, **center, **border, 'font_size': 6})
    fCenter10 = workbook.add_format({**wrap, **center, **border, 'font_size': 10})

    fCenterBold10 = workbook.add_format({**wrap, **center, **border, **bold, 'font_size': 10})
    fCenterBold11 = workbook.add_format({**wrap, **center, **border, **bold, 'font_size': 11})
    fCenterBold12 = workbook.add_format({**wrap, **center, **border, **bold, 'font_size': 12})
    fCenterBold16 = workbook.add_format({**wrap, **center, **border, **bold, 'font_size': 16})
    fCenterBold20 = workbook.add_format({**wrap, **center, **border, **bold, 'font_size': 20})

    def make_ticket(row, col, product_range="", code="", colour="", width=0, height=0, wall=True, floor=True, list_price=0, colours=[], sizes=[], material="PORCELAIN", slip_rating="", finishes=[], list_colour=""):
      tiles_per_sqm = 1000000/float(width)/float(height) # guess
      sqm_rounding = {} if (round(tiles_per_sqm, 2) == tiles_per_sqm) else {'num_format': '0.00'} # round to 2dp but use python's automatic rounding otherwise because xlsxwriter's is dodgy

      fancy_colour = (re.sub('(decor) ?', '', colour).strip()+' décor').upper() if ('decor' in colour) else colour.upper()

      if (d == "moments" and ("HOP MIX" in fancy_colour)): fancy_name = "HOP MIX"
      if (d == "parquet" and fancy_colour == "CHENE"): fancy_name = "CHENE (OAK)"

      worksheet.merge_range(row+0, col+0, row+2, col+6, 'BOYDEN TILES', fTitle)
      worksheet.merge_range(row+3, col+0, row+3, col+6, '2022', fCenter6)
      worksheet.merge_range(row+4, col+0, row+5, col+1, 'SOURCE', fCenterBold11)
      worksheet.merge_range(row+4, col+2, row+5, col+6, product_range, fCenterBold12)
      worksheet.merge_range(row+6, col+0, row+6, col+6, '', fBorder)
      worksheet.merge_range(row+7, col+0, row+8, col+1, 'REFERENCE', fCenterBold11)
      worksheet.merge_range(row+7, col+2, row+8, col+6, code, fCenterBold12)
      worksheet.merge_range(row+9, col+0, row+9, col+6, '', fBorder)
      worksheet.merge_range(row+10, col+0, row+11, col+1, 'COLOUR', fCenterBold11)
      worksheet.merge_range(row+10, col+2, row+11, col+6, fancy_colour, fCenterBold12)
      worksheet.merge_range(row+12, col+0, row+12, col+6, '', fBorder)
      worksheet.merge_range(row+13, col+0, row+14, col+1, 'SIZE', fCenterBold11)
      worksheet.merge_range(row+13, col+2, row+14, col+3, width, workbook.add_format({**wrap, **center, **bold, 'font_size': 20, 'bottom': 1, 'left': 1, 'top': 1}))
      worksheet.merge_range(row+13, col+4, row+14, col+4, 'X', workbook.add_format({**wrap, **center, **bold, 'font_size': 20, 'bottom': 1, 'top': 1}))
      worksheet.merge_range(row+13, col+5, row+14, col+6, height, workbook.add_format({**wrap, **center, **bold, 'font_size': 20, 'bottom': 1, 'right': 1, 'top': 1}))
      
      # dynamically generate formulas based on location
      sizeFormula = "=" + xlsxwriter.utility.xl_rowcol_to_cell(row+13, col+2) + "*" + xlsxwriter.utility.xl_rowcol_to_cell(row+13, col+5) + "/1000000" # =C14*F14/1000000
      exVATPriceFormula = "=ROUND(" + xlsxwriter.utility.xl_rowcol_to_cell(row+19, col+0) + "*" + xlsxwriter.utility.xl_rowcol_to_cell(row+16, col+3) + ", 2)" # =ROUND(A20*D17, 2)
      incVATPriceForumla = "=" + xlsxwriter.utility.xl_rowcol_to_cell(row+16, col+0) + "*1.2" # =A17*1.2
      perSQMFormula = "=1/" + xlsxwriter.utility.xl_rowcol_to_cell(row+15, col+0) # =1/A16

      worksheet.merge_range(row+15, col+0, row+15, col+6, sizeFormula, workbook.add_format({**wrap, 'left': 1, 'right': 1, 'font_color': 'white'}))
      worksheet.merge_range(row+16, col+0, row+17, col+2, exVATPriceFormula,workbook.add_format({**wrap, **center, **bold, **border, 'font_size': 16, 'num_format': '"£"#,##0.00'}))
      worksheet.merge_range(row+16, col+4, row+17, col+6, incVATPriceForumla, workbook.add_format({**wrap, **center, **bold, **border, 'font_size': 16, 'num_format': '"£"#,##0.00'}))
      worksheet.merge_range(row+18, col+0, row+18, col+2, 'PER MTR EX VAT', fCenter10)
      worksheet.merge_range(row+18, col+4, row+18, col+6, 'PER MTR INC VAT', fCenter10)
      worksheet.write(row+19, col+0, perSQMFormula, workbook.add_format({**wrap, **center, 'bottom': 1, 'left': 1, 'top': 1, 'font_size': 10, **sqm_rounding}))
      worksheet.merge_range(row+19, col+1, row+19, col+2, 'TILES PER MTR', workbook.add_format({**wrap, **center, 'bottom': 1, 'right': 1, 'top': 1, 'font_size': 10}))
      worksheet.merge_range(row+19, col+4, row+19, col+5, 'Wall', fCenter10)
      worksheet.merge_range(row+20, col+4, row+20, col+5, 'Floor', fCenter10)
      worksheet.merge_range(row+21, col+0, row+21, col+2, 'Available in:', fCenterBold10)

      for i in range(row+22, row+29):
        worksheet.merge_range(i, col+0, i, col+1, '', fCenter10)
        worksheet.write(i, col+2, '', fCenter10)
        worksheet.merge_range(i, col+4, i, col+5, '', fCenter10)
        worksheet.write(i, col+6, '', fCenter10)

      offset = 0
      # work out how many finishes this colour comes in
      colour_finishes_set = set([])
      for p in ranges[d]:
        if (list_colour in p[4].strip().lower()):
          colour_finishes_set.add(p[0]['Finish'].split('/')[0].strip().title())
      colour_finishes = list(colour_finishes_set)
      num_finishes = len(colour_finishes)

      # write colours on left
      for i in range(min(len(colours), 7)):
        worksheet.write(i+row+22, col, colours[i], fCenter10)
        tick_cell(i+row+22, col+2, -1.5)
      if (len(sizes) > 1):
        if (len(colours) + len(sizes) <= 7):
          # if there is space, write sizes on right anyway
          if ((not decor) and (len(finishes) <= 1)):
            for i in range(min(len(sizes), 4)):
              worksheet.write(i+row+24, col+4, sizes[i], fCenter10)
              tick_cell(i+row+24, col+6, -1.5)
          else:
            offset = 1
            sizes_len = min(len(sizes), 5)
            # write sizes on left
            for i in range(sizes_len):
              worksheet.write((7-sizes_len)+i+row+22, col, sizes[i], fCenter10)
              tick_cell((7-sizes_len)+i+row+22, col+2, -1.5)
        else:
          # if there is space, write sizes higher on right
          if ((not decor) and (len(finishes) <= 1)):
            for i in range(min(len(sizes), 5)):
              worksheet.write(i+row+24, col+4, sizes[i], fCenter10)
              tick_cell(i+row+24, col+6, -1.5)
          else:
            # write sizes on right
            for i in range(min(len(sizes), 5)):
              worksheet.write(i+row+26, col+4, sizes[i], fCenter10)
              tick_cell(i+row+26, col+6, -1.5)
      
      # if decor add Plain/Decor else add Finishes
      # offset to space all nicely
      if (decor):
        worksheet.write(row+24+offset, col+4, 'Plain', fCenter10)
        tick_cell(row+24+offset, col+6, -1.5)
        worksheet.write(row+25+offset, col+4, 'Décor', fCenter10)
        tick_cell(row+25+offset, col+6, -1.5)
      else:
        if (len(finishes) > 1):
          for i, f in enumerate(colour_finishes):
            worksheet.write(i+row+24+offset, col+4, f, fCenter10)
            tick_cell(i+row+24+offset, col+6, -1.5)

      # write material on top right
      worksheet.write(row+22, col+4, material, fCenter10)
      tick_cell(row+22, col+6, -1.5)

      # if slip rating write to bottom right
      if (str(slip_rating) != 'nan'):
        worksheet.write(row+28, col+4, 'Rating', fCenter10)
        worksheet.write(row+28, col+6, slip_rating, fCenter10)

      worksheet.write(row+16, col+3, list_price, workbook.add_format({'font_color': 'white'}))
      worksheet.write(row+19, col+6, '', fBorder)
      if floor or wall: tick_cell(row+19, col+6)
      worksheet.write(row+20, col+6, '', fBorder)
      if floor: tick_cell(row+20, col+6)
      worksheet.write(row+20, col+0, '', workbook.add_format({'left': 1}))
      worksheet.write(row+21, col+6, '', workbook.add_format({'right': 1}))
      worksheet.write(row+28, col+3, '', workbook.add_format({'bottom': 1}))
      
      worksheet.set_column_pixels(col+0, col+6, 45)
      row_heights = [16, 16, 16, 9, 5, 16, 9, 16, 5, 9, 16, 5, 9, 16, 18, 16, 16, 16, 22, 22, 22, 22, 20, 20, 20, 20, 20, 20, 20]

      for i in range(0, 28):
        worksheet.set_row_pixels(row+i, row_heights[i])

    worksheet.set_column_pixels(7, 7, 18)
    worksheet.set_row_pixels(29, 23)
    
    if (len(sheet)>0): make_ticket(0, 0, d.upper(), sheet[0][0]['Product Code'], sheet[0][1], int(sheet[0][2]), int(sheet[0][3]), 'Wall' in sheet[0][0]['Application'], 'Floor' in sheet[0][0]['Application'], float(sheet[0][0]['List Price']), info[d]['colours'], info[d]['sizes'], 'CERAMIC' if sheet[0][0]['Material'] == 'Ceramic' else 'PORCELAIN', sheet[0][0]['Slip Rating'], info[d]['finishes'], sheet[0][4])
    if (len(sheet)>1): make_ticket(0, 8, d.upper(), sheet[1][0]['Product Code'], sheet[1][1], int(sheet[1][2]), int(sheet[1][3]), 'Wall' in sheet[1][0]['Application'], 'Floor' in sheet[1][0]['Application'], float(sheet[1][0]['List Price']), info[d]['colours'], info[d]['sizes'], 'CERAMIC' if sheet[1][0]['Material'] == 'Ceramic' else 'PORCELAIN', sheet[1][0]['Slip Rating'], info[d]['finishes'], sheet[1][4])
    if (len(sheet)>2): make_ticket(30, 0, d.upper(), sheet[2][0]['Product Code'], sheet[2][1], int(sheet[2][2]), int(sheet[2][3]), 'Wall' in sheet[2][0]['Application'], 'Floor' in sheet[2][0]['Application'], float(sheet[2][0]['List Price']), info[d]['colours'], info[d]['sizes'], 'CERAMIC' if sheet[2][0]['Material'] == 'Ceramic' else 'PORCELAIN', sheet[2][0]['Slip Rating'], info[d]['finishes'], sheet[2][4])
    if (len(sheet)>3): make_ticket(30, 8, d.upper(), sheet[3][0]['Product Code'], sheet[3][1], int(sheet[3][2]), int(sheet[3][3]), 'Wall' in sheet[3][0]['Application'], 'Floor' in sheet[3][0]['Application'], float(sheet[3][0]['List Price']), info[d]['colours'], info[d]['sizes'], 'CERAMIC' if sheet[3][0]['Material'] == 'Ceramic' else 'PORCELAIN', sheet[3][0]['Slip Rating'], info[d]['finishes'], sheet[3][4])

    workbook.close()