from earning_server import EServer

e_server = EServer()

while (True):
  res = e_server.query_EServer()
  if res == 'exit':
    break

while (True):
  stock = input("Stock for eanring server to check: ")
  if stock == 'exit':
    break
  res = e_server.check(stock)
  print(f'check() in earning server returns: {res}')
