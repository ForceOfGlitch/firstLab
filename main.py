import pandas as pd
from apyori import apriori
import time

start_time = time.time()

dataset = pd.read_csv("Market.csv")

'''
Поддержка — это вероятность того, что событие произойдет, количество транзакций, включающих A, деленное на общее количество транзакций.
Доверие - вероятность наступления события А при условии, что событие В уже произошло.
Лифт - вероятность того, что товар будет куплен при покупке другого товара, при этом учитывается популярность обоих товаров. 
'''

#Преобразуем фрейм данных Pandas в список списков
transactions = []
for i in range(0, 1897):
 transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

rules = apriori(transactions=transactions, min_support=0.001, min_cinfidence=0.2, min_lift=3, min_length=2,
                 max_length=2)


# Выводим списком количество правил
results = list(rules)

def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]  # сохраняем первый товар из всех результатов
    rhs = [tuple(result[2][0][1])[0] for result in results]  # из lhs получаем второй товар, покупаемый после первого
    # сохрвняем результаты поддержки, доверия и лифта
    supports = [result[1] for result in results]
    confidences =[result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
# сохраняем все переменные в одном фрейме данных
resultsinDataFrame = pd.DataFrame(inspect(results), columns =
    ["Left hand side", "Right hand side", "Support", "Confidence", "Lift"])

print(resultsinDataFrame.nlargest(n = 30, columns = "Support"))  # сортируем конечные результаты в порядке убывания значений
#print(inspect(results).__len__())
print(results.__len__())



print("--- %s seconds ---" % (time.time() - start_time))


