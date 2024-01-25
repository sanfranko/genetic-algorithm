### Импорт необходимых модулей:

```python
// Python
import random
import matplotlib.pyplot as plt
```
В этой части кода импортируются модули random для генерации случайных чисел и matplotlib.pyplot для построения графиков.



### Константы генетического алгоритма:

```python
// Python
POPULATION_SIZE = 100  # количество индивидуумов в популяции
MAX_GENERATIONS = 200  # максимальное количество поколений
P_CROSSOVER = 0.9  # вероятность скрещивания
P_MUTATION = 0.1  # вероятность мутации индивидуума
N_VECTOR = 2  # количество генов в хромосоме
LIMIT_VALUE_TOP = 100
LIMIT_VALUE_DOWN = -100
RANDOM_SEED = 1
random.seed(RANDOM_SEED)
```
Здесь определены константы, используемые в генетическом алгоритме, такие как размер популяции, максимальное количество поколений, вероятности скрещивания и мутации, а также другие параметры для создания и оценки индивидуумов.



### Класс Individual:

```python
// Python
class Individual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.value = 0
```
Этот класс представляет индивидуума популяции. У него есть свойство value, которое представляет его значение в функции приспособленности.



### Функция приспособленности fitness_function:

```python
// Python
def fitness_function(f):
    return f[0] ** 2 + 1.5 * f[1] ** 2 - 2 * f[0] * f[1] + 4 * f[0] - 8 * f[1]
```
Это функция приспособленности, которую мы пытаемся оптимизировать в генетическом алгоритме.



### Функции для создания индивидуумов и популяции:

```python
// Python
def individualCreator():
    return Individual([random.randint(LIMIT_VALUE_DOWN, LIMIT_VALUE_TOP) for i in range(N_VECTOR)])

def populationCreator(n=0):
    return list([individualCreator() for i in range(n)])
```
Эти функции создают индивидуумов и популяцию, используя предыдуще определенные параметры.



### Создание начальной популяции:

```python
// Python
population = populationCreator(n=POPULATION_SIZE)
```



### Вычисление значений функции приспособленности для индивидуумов популяции:

```python
// Python
fitnessValues = list(map(fitness_function, population))
for individual, fitnessValue in zip(population, fitnessValues):
    individual.value = fitnessValue
```
Здесь вычисляются значения функции приспособленности для каждого индивидуума в популяции.



### Сортировка популяции по значению функции приспособленности:

```python
// Python
population.sort(key=lambda ind: ind.value)
```



### Создание списка для хранения значений приспособленности:

```python
// Python
MinFitnessValues = []
meanFitnessValues = []
BadFitnessValues = []
```
Эти списки будут использоваться для отслеживания минимального, среднего и максимального значения приспособленности в каждом поколении.



### Клонирование индивидуума:

```python
// Python
def clone(value):
    ind = Individual(value[:])
    ind.value = value.value
    return ind
```



### Метод выбора (селекции):

```python
// Python
def selection(popula, n=POPULATION_SIZE):
    offspring = []
    for i in range(n):
        i1 = i2 = i3 = i4 = 0
        while i1 in [i2, i3, i4] or i2 in [i1, i3, i4] or i3 in [i1, i2, i4] or i4 in [i1, i2, i3]:
            i1, i2, i3, i4 = random.randint(0, n - 1), random.randint(0, n - 1), random.randint(0, n - 1), random.randint(0, n - 1)

        offspring.append(min([popula[i1], popula[i2], popula[i3], popula[i4]], key=lambda ind: ind.value))
    return offspring
```
Этот метод выбирает индивидуумов для создания следующего поколения.


### Мутация:

```python
// Python
def mutation(mutant, indpb=0.04, percent=0.05):
    for index in range(len(mutant)):
        if random.random() < indpb:
            mutant[index] += random.randint(-1, 1) * percent * mutant[index]
```
Эта функция выполняет мутацию у индивидуума с определенной вероятностью.


### Основной цикл генетического алгоритма:

```python
// Python
while generationCounter < MAX_GENERATIONS:
    generationCounter += 1
    # Выполнение выбора
    offspring = selection(population)
    offspring = list(map(clone, offspring))
    # Скрещивание
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < P_CROSSOVER:
            crossbreeding(child1, child2)
    # Мутация
    for mutant in offspring:
        if random.random() < P_MUTATION:
            mutation(mutant, indpb=1.0 / N_VECTOR)
    # Обновление значений приспособленности
    freshFitnessValues = list(map(fitness_function, offspring))
    for individual, fitnessValue in zip(offspring, freshFitnessValues):
        individual.value = fitnessValue
    # Замена популяции новым поколением
    population[:] = offspring
    fitnessValues = [ind.value for ind in population]
    # Отслеживание статистики
    minFitness = min(fitnessValues)
    meanFitness = sum(fitnessValues) / len(population)
    maxFitness = max(fitnessValues)
    MinFitnessValues.append(minFitness)
    meanFitnessValues.append(meanFitness)
    BadFitnessValues.append(maxFitness)
    # Вывод статистики о приспособленности лучшего индивидуума
    print(f"Поколение {generationCounter}: Функция приспособленности = {minFitness}, Средняя приспособленность = {meanFitness}")
    best_index = fitnessValues.index(min(fitnessValues))
    print("Лучший индивидуум =", *population[best_index], "\n")
```
Этот блок кода представляет основной цикл генетического алгоритма, в котором выполняются операции выбора, скрещивания, мутации, обновления значений приспособленности и замены текущей популяции на новое поколение.



### Отображение графика:

```python
// Python
plt.plot(MinFitnessValues[int(MAX_GENERATIONS * 0.1):], color='red')
plt.plot(meanFitnessValues[int(MAX_GENERATIONS * 0.1):], color='green')
plt.plot(BadFitnessValues[int(MAX_GENERATIONS * 0.1):], color='blue')
plt.xlabel('Поколение')
plt.ylabel('Мин/средняя/max приспособленность')
plt.title('Зависимость min, mean, max приспособленности от поколения')
plt.show()
```

В этой части кода строятся графики, отображающие изменение минимального, среднего и максимального значения приспособленности в зависимости от поколения с помощью библиотеки Matplotlib.

