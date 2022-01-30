import xlsxwriter

workbook = xlsxwriter.Workbook('ticket.xlsx')
worksheet = workbook.add_worksheet('Sheet1')

border = {'border': 1}
center = {'align': 'center', 'valign': 'vcenter'}
bold = {'bold': True}

def tick_cell(row, col):
  tick_scale = 1/3
  tick_format = {'x_scale': tick_scale, 'y_scale': tick_scale, 'x_offset': 10, 'y_offset': -6}
  worksheet.insert_image(row, col, 'tick.png', tick_format)

fTitle = workbook.add_format({**center, **border, **bold, 'font_name': 'Aparajita', 'font_size': 26, 'font_color': 'white', 'bg_color': 'black'})

fBorder = workbook.add_format(border)

fCenter6 = workbook.add_format({**center, **border, 'font_size': 6})
fCenter10 = workbook.add_format({**center, **border, 'font_size': 10})

fCenterBold10 = workbook.add_format({**center, **border, **bold, 'font_size': 10})
fCenterBold11 = workbook.add_format({**center, **border, **bold, 'font_size': 11})
fCenterBold12 = workbook.add_format({**center, **border, **bold, 'font_size': 12})
fCenterBold16 = workbook.add_format({**center, **border, **bold, 'font_size': 16})
fCenterBold20 = workbook.add_format({**center, **border, **bold, 'font_size': 20})

def make_ticket(row, col, product_range="", code="", colour="", width=0, height=0, wall=True, floor=True, list_price=0, colours=[], sizes=[], material="PORCELAIN", slip_rating=""):
  tiles_per_sqm = 100.00
  sqm_rounding = {} if (round(tiles_per_sqm, 2) == tiles_per_sqm) else {'num_format': '0.00'} # round to 2dp but use python's automatic rounding otherwise because xlsxwriter's is dodgy

  worksheet.merge_range(row+0, col+0, row+2, col+6, 'BOYDEN TILES', fTitle)
  worksheet.merge_range(row+3, col+0, row+3, col+6, '2022', fCenter6)
  worksheet.merge_range(row+4, col+0, row+5, col+1, 'SOURCE', fCenterBold11)
  worksheet.merge_range(row+4, col+2, row+5, col+6, 'RANGE', fCenterBold12)
  worksheet.merge_range(row+6, col+0, row+6, col+6, '', fBorder)
  worksheet.merge_range(row+7, col+0, row+8, col+1, 'REFERENCE', fCenterBold11)
  worksheet.merge_range(row+7, col+2, row+8, col+6, 'CODE', fCenterBold12)
  worksheet.merge_range(row+9, col+0, row+9, col+6, '', fBorder)
  worksheet.merge_range(row+10, col+0, row+11, col+1, 'COLOUR', fCenterBold11)
  worksheet.merge_range(row+10, col+2, row+11, col+6, 'COLOUR', fCenterBold12)
  worksheet.merge_range(row+12, col+0, row+12, col+6, '', fBorder)
  worksheet.merge_range(row+13, col+0, row+14, col+1, 'SIZE', fCenterBold11)
  worksheet.merge_range(row+13, col+2, row+14, col+3, '100', workbook.add_format({**center, **bold, 'font_size': 20, 'bottom': 1, 'left': 1, 'top': 1}))
  worksheet.merge_range(row+13, col+4, row+14, col+4, 'X', workbook.add_format({**center, **bold, 'font_size': 20, 'bottom': 1, 'top': 1}))
  worksheet.merge_range(row+13, col+5, row+14, col+6, '100', workbook.add_format({**center, **bold, 'font_size': 20, 'bottom': 1, 'right': 1, 'top': 1}))
  worksheet.merge_range(row+15, col+0, row+15, col+6, '=C14*F14/1000000', workbook.add_format({'left': 1, 'right': 1, 'font_color': 'white'}))
  worksheet.merge_range(row+16, col+0, row+17, col+2, '=ROUND(A20*D17, 2)', workbook.add_format({**center, **bold, **border, 'font_size': 16, 'num_format': '"£"#,##0.00'}))
  worksheet.merge_range(row+16, col+4, row+17, col+6, '=A17*1.2', workbook.add_format({**center, **bold, **border, 'font_size': 16, 'num_format': '"£"#,##0.00'}))
  worksheet.merge_range(row+18, col+0, row+18, col+2, 'PER MTR EX VAT', fCenter10)
  worksheet.merge_range(row+18, col+4, row+18, col+6, 'PER MTR INC VAT', fCenter10)
  worksheet.write(row+19, col+0, '=1/A16', workbook.add_format({**center, **border, 'font_size': 10, **sqm_rounding}))
  worksheet.merge_range(row+19, col+1, row+19, col+2, 'TILES PER MTR', fCenter10)
  worksheet.merge_range(row+19, col+4, row+19, col+5, 'Wall', fCenter10)
  worksheet.merge_range(row+20, col+4, row+20, col+5, 'Floor', fCenter10)
  worksheet.merge_range(row+21, col+0, row+21, col+2, 'Available in:', fCenterBold10)

  for i in range(row+22, row+29):
    worksheet.merge_range(i, col+0, i, col+1, '', fCenter10)
    worksheet.write(i, col+2, '', fCenter10)
    worksheet.merge_range(i, col+4, i, col+5, '', fCenter10)
    worksheet.write(i, col+6, '', fCenter10)

  worksheet.write(row+16, col+3, '0.1', workbook.add_format({'font_color': 'white'}))
  worksheet.write(row+19, col+6, '', fBorder)
  if wall: tick_cell(row+19, col+6)
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
make_ticket(0, 0)
make_ticket(0, 8)
make_ticket(30, 0, floor=False, material="CERAMIC")
make_ticket(30, 8)

workbook.close()