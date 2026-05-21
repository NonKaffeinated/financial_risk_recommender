from earning_server import EServer

e_server = EServer()

while (True):
  res = e_server.query_EServer()
  if res == 'exit':
    break

while (True):
  stock = input("Stock for earning server to check: ")
  if stock == 'exit':
    break
  res = e_server.check(stock)
  print(f'check() in earning server returns: {res}')

if res[1] == 1:
  stock = "NVDA"
  query = f"Does Bank of America recommend individual investor to buy, sell, or hold {stock} stock before earning release date on May 20, 2026?"
  e_server_response = e_server.query_EServer_query(query)
  print(f'Final Earning Server Response:\n\n{e_server_response}\n')