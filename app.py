import streamlit as st
import cv2
import numpy as np
import re

from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance
from sympy import symbols, sympify, solve

# Определим константы
#THRESHOLD_VALUE = 117                                    # Пороговое значение яркости
W_min, H_min = 14, 14                                    # Минимальные размеры контура
fontScale = 1                                            # Размер шрифта
new_width = 1400                                         # Ширина обработанного изображения

st.title('Проверка письменных работ по математике')

THRESHOLD_VALUE = st.number_input("Введите порог яркости", min_value=111, max_value=200, step=1)

output_recognized_characters = st.checkbox("Выводить распознанные символы")
show_found_degrees = st.checkbox("Показать найденные степени")
show_outlines = st.checkbox("Показать контуры найденных символов")
show_rows = st.checkbox("Выводить результаты распознавания блоков")
dont_cut = st.checkbox("Не пытаться делить контуры")
show_eq = st.checkbox("Показать найденные знаки равенства")

filename = st.file_uploader('Загрузите картинку с решением', type=['jpg'])

model_28_28  = load_model('model_28_28.h5')
model_56_28  = load_model('model_56_28.h5')
model_112_28 = load_model('model_112_28.h5')

# Размеры картинки для распознавания
IMG_WIDTH, IMG_HEIGHT = 28, 28

def recognition(image, relationship_between_parties):

  # Определение новых размеров холста
  new_size = max(image.shape[0], image.shape[1])

  # Определение сдвига по горизонтали и вертикали
  diff_width = new_size - image.shape[1]
  diff_height = new_size // relationship_between_parties - image.shape[0]

  # Расширение изображения
  top = int(diff_height / 2)
  bottom = diff_height - top
  left = int(diff_width / 2)
  right = diff_width - left
  expanded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)

  image = cv2.resize(expanded_image, (28 * relationship_between_parties, 28))

  img_np = np.array(image)                # Перевод в numpy-массив

  # Преобразование x_train в тип float32 (числа с плавающей точкой) и нормализация
  img_np = img_np.astype('float32') / 255.

  digit = np.expand_dims(img_np, axis=0)

  s1 = '0123456789ABCDEFGH{xK<mN>PбRSTUVwxyzabdefghnqrt+'

  ch = ""
  v = 0

  # Распознавание примера
  if relationship_between_parties == 1:
    prediction = model_28_28.predict(digit, verbose = 0)
    if max(prediction[0]) > 0.3:
      v = max(prediction[0])
      number = np.argmax(prediction[0])
      if number <= 47:
        ch = s1[number]
      elif number == 48:
        ch = "Answer"
      elif number == 49:
        ch = "sqrt"
      elif number == 50:
        ch = "("
      elif number == 51:
        ch = ")"
      elif number == 52:
        ch = "["
  elif relationship_between_parties == 2:
    prediction = model_56_28.predict(digit, verbose = 0)
    if max(prediction[0]) > 0.5:
      v = max(prediction[0])
      number = np.argmax(prediction[0])
      if number == 0:
        ch = "-"
      elif number == 1:
        ch = "sqrt"
      elif number == 2:
        ch = "Answer"
      elif number == 3:
        ch = "No"
      elif number == 4:
        ch = "roots"
      elif number == 5:
        ch = "Examination"
      elif number == 6:
        ch = "x"
  else:
    prediction = model_112_28.predict(digit, verbose = 0)
    if max(prediction[0]) > 0.5:
      v = max(prediction[0])
      number = np.argmax(prediction[0])
      if number == 0:
        ch = "-"
      elif number == 1:
        ch = "sqrt"
      elif number == 2:
        ch = "Independent"
      elif number == 3:
        ch = "we get"
      elif number == 4:
        ch = "let's multiply"
      elif number == 5:
        ch = "equations"
      elif number == 6:
        ch = "members"

  return ch, v

def crop_black_borders(img):

    rows = img.shape[0]
    top = 0
    bottom = rows - 1

    while top < rows:
      if any(img[top] != 0):
        break
      top += 1

    while bottom >= 0:
      if any(img[bottom] != 0):
        break
      bottom -= 1

    if top <= bottom:
        return img[top:bottom+1, :]
    else:
        return img

def index_symbol_bottom(m):
  dy_min = 1000
  c = -1
  for j in range(len(rez)):
    if m == j:
      continue
    if (rez[m][0] < rez[j][2] and rez[m][2] > rez[j][0]) or (rez[j][0] < rez[m][2] and rez[j][2] > rez[m][0]):
      dy = rez[j][1] - rez[m][3]
      if dy < dy_min and rez[j][1] > rez[m][1]:
        dy_min = dy
        c = j

  return c

def index_symbol_above(m):
  dy_min = 1000
  c = -1
  for j in range(len(rez)):
    if m == j:
      continue
    if (rez[m][0] < rez[j][2] and rez[m][2] > rez[j][0]) or (rez[j][0] < rez[m][2] and rez[j][2] > rez[m][0]):
      dy = rez[m][1] - rez[j][3]
      if dy < dy_min and rez[m][1] > rez[j][1]:
        dy_min = dy
        c = j

  return c

def index_symbol_right(i):
  dx_min = 1000
  c = -1
  for j in range(len(rez)):
    if i == j:
      continue
    if rez[j][0] >= rez[i][0] and rez[j][2] >= rez[i][2] and rez[j][1] < rez[i][3] and rez[j][3] > rez[i][1]:
      dx = rez[j][0] - rez[i][0]
      if dx < dx_min:
        dx_min = dx
        c = j
  return c

def find_drob():
  number = -1
  for i in arr:
    number += 1
    for j in range(len(i[4])):
      if i[4][j][4] == '-':
        there_is_top = False
        for k in range(len(i[4])):
          if k == j:
            continue
          if i[4][k][0] < i[4][j][2] and i[4][k][2] > i[4][j][0] and i[4][k][1] < i[4][j][1]:
            there_is_top = True
            break
        there_is_down = False
        for k in range(len(i[4])):
          if k == j:
            continue
          if i[4][k][0] < i[4][j][2] and i[4][k][2] > i[4][j][0] and i[4][k][3] > i[4][j][3]:
            there_is_down = True
            break
        if (there_is_top == True and there_is_down == False):
          return number
        if (there_is_top == False and there_is_down == True):
          return number + 1
  return number

def find_word_in_area(x1, y1, x2, y2):
  arr_symb = []
  indexes = []
  for i in range(len(rez)):
    if rez[i][0] >= x1 and rez[i][2] <= x2 and rez[i][3] <= y2 and rez[i][1] >= y1:
      arr_symb.append(rez[i])
      indexes.append(i)

  for i in range(len(arr_symb) - 1):
    for j in range(i + 1, len(arr_symb)):
      if arr_symb[i][0] > arr_symb[j][0]:
        arr_symb[i], arr_symb[j] = arr_symb[j], arr_symb[i]
  s = ""
  for i in arr_symb:
    s += i[4]

  return s, indexes

# Распознавание блока
def block_recognition(block):
  for i in range(len(block[4]) - 1):
    for j in range(i + 1, len(block[4])):
      if block[4][i][0] > block[4][j][0]:
        block[4][i], block[4][j] = block[4][j], block[4][i]

  ind1, ind2 = [], []
  for i in range(len(block[4])):
    if block[4][i][4] == "-":
      numerator, ind1   = find_word_in_area(block[4][i][0], block[1], block[4][i][2], block[4][i][1])
      if numerator == '-':
        continue
      denominator, ind2 = find_word_in_area(block[4][i][0], block[4][i][3], block[4][i][2], block[3])
      if denominator == "-":
        continue
      if numerator != "" and denominator != "":
        block[4][i][4] = "(" + numerator + ") / (" + denominator + ")"
        for j in ind1:
          if block[4][i][0] > rez[j][0]:
            block[4][i][0] = rez[j][0]
          if block[4][i][1] > rez[j][1]:
            block[4][i][1] = rez[j][1]
          if block[4][i][2] < rez[j][2]:
            block[4][i][2] = rez[j][2]
          if block[4][i][3] < rez[j][3]:
            block[4][i][3] = rez[j][3]
          for k in range(len(block[4])):
            if block[4][k] == rez[j]:
              block[4][k][4] = ""
        for j in ind2:
          if block[4][i][0] > rez[j][0]:
            block[4][i][0] = rez[j][0]
          if block[4][i][1] > rez[j][1]:
            block[4][i][1] = rez[j][1]
          if block[4][i][2] < rez[j][2]:
            block[4][i][2] = rez[j][2]
          if block[4][i][3] < rez[j][3]:
            block[4][i][3] = rez[j][3]
          for k in range(len(block[4])):
            if block[4][k] == rez[j]:
              block[4][k][4] = ""

  s = block[4][0][4]
  for i in range(1, len(block[4])):
    if block[4][i][3] < (block[4][i - 1][1] + block[4][i - 1][3]) / 2 and block[4][i][4] != '-':
      s = s + " * * "
    elif block[4][i - 1][4] in '23456789' and (block[4][i][4] == 'x' or block[4][i][4] == '('):
      s = s + " * "
    elif block[4][i - 1][4] == ")" and block[4][i][4] == "(":
      s = s + " * "
    elif block[4][i - 1][4] == "x" and block[4][i][4] == "(":
      s = s + " * "

    s += block[4][i][4]

  s = s.replace("* * =", "=")
  s = s.replace("* *  =", "=")
  s = s.replace("- * *", "-")
  s = s.replace("* *  * *  * * -", "-")
  s = s.replace("* *  * * -", "-")
  s = s.replace("* *  =", "=")
  if s[-2:] == "01":
      s = s[:-1]
  if s[-5:] == "* * (":
    s = s[:-5]

  # if s[-2:] == "01" or s[-2:] == "0q" or s[-2:] == "0g":
  #   s = s[:-1]

  s = remove_redundant_parentheses(s)

  return s

def remove_redundant_parentheses(s):
    pattern = r'\((.)\)'
    while True:
        match = re.search(pattern, s)
        if not match:
            break
        s = s[:match.start()] + match.group(1) + s[match.end():]
    return s

def evaluate_expressions(s):
    pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)=(-?\d+(\.\d*)?([eE][+-]?\d+)?)'
    while True:
        match = re.search(pattern, s)
        if not match:
            break
        var_name = match.group(1)
        expression = match.group(2)
        value = eval(expression)
        s = re.sub(pattern, f'{var_name}={value}', s, count=1)
    return s

if not filename is None:

    image = Image.open(filename)
    enhancer = ImageEnhance.Contrast(image)
    image_new = enhancer.enhance(2)
    #st.image(image_new)
    image_new = image_new.save("img.jpg")
    im = cv2.imread("img.jpg")

    # Изменение размеров изображения
    height, width = im.shape[:2]
    new_height = int((new_width / width) * height)
    im = cv2.resize(im, (new_width, new_height))

    # Переводим изображение в оттенки серого
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Инвертируем цвета (черное становится белым и наоборот)
    imgray = cv2.bitwise_not(imgray)

    # Пороговое преобразование изображения
    ret, thresh = cv2.threshold(imgray, THRESHOLD_VALUE, 255, 0)

    # Поиск контуров, их распознавание и запись результатов в массив rez
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rez = []

    for index_cnt in range(len(contours)):

        x, y, w, h = cv2.boundingRect(contours[index_cnt])
        if w < W_min and h < H_min:
            continue

        # Создадим маску, используя контур
        mask = np.zeros(thresh.shape, dtype='uint8')
        cv2.drawContours(mask, contours, index_cnt, 255, -1)

        # Вырежем контур из изображения
        cropped = cv2.bitwise_and(thresh, mask)

        image = cropped[y:y + h, x:x + w]

        if w < 2 * h:
            result, v = recognition(image, 1)
            if dont_cut != True and (result == 'F' or result == 'D' or result == 'U' or result == '8' or result == 'a' or result == 'G' or result == 'd'):
                img1 = crop_black_borders(cropped[y:y + h, x:x + w // 2 + 2])
                result1, v1 = recognition(img1, 1)
                img2 = crop_black_borders(cropped[y:y + h, x + w // 2:x + w])
                result2, v2 = recognition(img2, 1)
                if (v1 + v2) / 2 > v:
                    result = result1 + result2
        elif w < 4 * h:
            result, v = recognition(image, 2)
        else:
            result, v = recognition(image, 4)

        if result != "" and result != "[":
            rez.append([x, y, x + w, y + h, result])
            if output_recognized_characters:
                cv2.putText(im, result, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 0, 0), 2)
            if show_outlines:
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 200, 255), 1)

    del_elem = []
    for i in rez:
        for j in rez:
            if i == j:
                continue
            if i[0] <= j[0] <= j[2] <= i[2] and i[1] <= j[1] <= j[3] <= i[3]:
                del_elem.append(j)

    for i in del_elem:
        if i in rez:
            rez.remove(i)

    del_elem = []
    for i in range(len(rez)):
        if rez[i][4] != '-':
            continue
        index = index_symbol_above(i)
        if index > -1 and rez[index][4] == '-':
            # cv2.rectangle(im, (rez[i][0], rez[i][1]), (rez[i][2], rez[i][3]), (100, 255, 225), 1)

            if index_symbol_bottom(index) == i and max(rez[i][2], rez[index][2]) - min(rez[i][0], rez[index][0]) > max(rez[i][3], rez[index][3]) - min(rez[i][1], rez[index][1]):
                rez[i][4] = '='
                rez[i][0] = min(rez[i][0], rez[index][0])
                rez[i][1] = min(rez[i][1], rez[index][1])
                rez[i][2] = max(rez[i][2], rez[index][2])
                rez[i][3] = max(rez[i][3], rez[index][3])
                rez[index][4] = '='
                del_elem.append(rez[index])
                if show_eq:
                    cv2.rectangle(im, (rez[i][0], rez[i][1]), (rez[i][2], rez[i][3]), (255, 255, 100), 1)

    for i in range(len(rez)):
        if rez[i][4] != "=":
            continue
        index = index_symbol_right(i)
        if index > -1 and rez[index][4] == ">":
            rez[i][4] = "=>"
            rez[i][0] = min(rez[i][0], rez[index][0])
            rez[i][1] = min(rez[i][1], rez[index][1])
            rez[i][2] = max(rez[i][2], rez[index][2])
            rez[i][3] = max(rez[i][3], rez[index][3])
            del_elem.append(rez[index])

    for i in del_elem:
        if i in rez:
            rez.remove(i)

    # Поиск степеней
    del_elem = []
    elem_degrees = []
    for i in rez:
        for j in rez:
            if i == j or i[4] == '-' or j[4] == '-' or i[4] == 'Answer' or j[4] == 'Answer' or i[
                4] == 'Examination' or j[4] == 'Examination' or i[4] == '=' or j[4] == '=' or i[4] == 'roots' or j[
                4] == 'roots' or i[4] == 'members' or j[4] == 'members' or (i[4] == '(' and i[3] > j[3]) or (j[
                4] == '(' and j[3] > i[3]) or (i[4] == ')' and i[3] < j[3]) or (j[4] == ')' and j[3] < i[3]):
                continue
            if j[0] > i[2] and j[0] < 2 * i[2] - i[0] and j[3] < (i[1] + i[3]) / 2 and j[3] > i[1] - (i[3] - i[1]) / 2:
                elem_degrees.append([i, j])

    for i in elem_degrees:
        for j in elem_degrees:
            if i == j:
                continue
            if i[1] == j[1]:
                if i[0][0] < j[0][0]:
                    elem_degrees.remove(i)
                else:
                    elem_degrees.remove(j)

    for i in elem_degrees:
        i[0][4] += " * * " + i[1][4]
        i[0][1] = i[1][1]
        i[0][2] = i[1][2]
        del_elem.append(i[1])
        if show_found_degrees:
            cv2.rectangle(im, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (255, 255, 0), 1)

    for i in del_elem:
        if i in rez:
            rez.remove(i)

    arr = []  # Массив блоков
    for i in range(0, len(rez)):
        if rez[i][4] == 'roots':
            continue
        found_group = False
        for j in arr:
            if rez[i][1] <= j[3] and rez[i][3] >= j[1]:
                if rez[i][0] < j[0]:
                    j[0] = rez[i][0]
                if rez[i][1] < j[1]:
                    j[1] = rez[i][1]
                if rez[i][2] > j[2]:
                    j[2] = rez[i][2]
                if rez[i][3] > j[3]:
                    j[3] = rez[i][3]
                j[4].append(rez[i])
                found_group = True
        if not found_group:
            arr.append([rez[i][0], rez[i][1], rez[i][2], rez[i][3], [rez[i]]])

    del_elem = []
    for i in arr:
        for j in arr:
            if i == j:
                continue
            if i[0] <= j[0] <= j[2] <= i[2] and i[1] <= j[1] <= j[3] <= i[3]:
                del_elem.append(j)

    for i in del_elem:
        if i in arr:
            arr.remove(i)

    for i in range(0, len(rez)):
        for j in arr:
            if rez[i][1] >= j[1] and rez[i][3] <= j[3] and rez[i][0] >= j[0] and rez[i][2] <= j[2] and not (
                    rez[i] in j[4]):
                j[4].append(rez[i])

    arr = list(reversed(arr))

    i = find_drob()
    if i > -1 and i < len(arr) - 1:
        if arr[i + 1][0] < arr[i][0]:
            arr[i][0] = arr[i + 1][0]
        if arr[i + 1][2] > arr[i][2]:
            arr[i][2] = arr[i + 1][2]
        arr[i][3] = arr[i + 1][3]
        for j in arr[i + 1][4]:
            arr[i][4].append(j)
        arr.pop(i + 1)

    for i in arr:
        cv2.rectangle(im, (i[0], i[1]), (i[2], i[3]), (255, 255, 200), 1)

    x = symbols('x')

    a, b, c = 0, 0, 0  # Коэффициенты найденного квадратного уравнения

    answer = ""
    for i in arr:
        s = block_recognition(i)
        in_s = s
        print(in_s)
        if s[0] == 'D':
            pattern = r'=(.*?)(?:=|<|>|$)'
            match = re.search(pattern, s)
            if match:
                expression = match.group(1)
                try:
                    value = eval(expression)
                    s = f"D = {value}"
                except Exception as e:
                    pass
        elif s[0:2] == 'x1':
            pattern = r'=(.*?)(?:=|<|>|$)'
            match = re.search(pattern, s)
            if match:
                expression = match.group(1)
                try:
                    value = eval(expression)
                    s = f"x1 = {value}"
                    if value != x1 and value != x2:
                        cv2.rectangle(im, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 2)
                except Exception as e:
                    pass
        elif s[0:2] == 'x2':
            pattern = r'=(.*?)(?:=|<|>|$)'
            match = re.search(pattern, s)
            if match:
                expression = match.group(1)
                try:
                    value = eval(expression)
                    s = f"x2 = {value}"
                    if value != x1 and value != x2:
                        cv2.rectangle(im, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 2)
                except Exception as e:
                    pass
        else:
            s = s.replace("3x", "3 * x")
            s = s.replace("* *  =", "=")

            left_part = ""
            right_part = ""
            index_eq = s.find("=")
            r = []
            x1, x2 = 0, 0
            if index_eq > -1:
                left_part = s[:index_eq]
                right_part = s[index_eq + 1:]
                try:
                    if right_part != '0':
                        equation = sympify(left_part + "-(" + right_part + ")")
                    else:
                        equation = sympify(left_part)
                    s = solve(equation, x)
                    for j in s:
                        if str(j).find('I') == -1:
                            r.append(j)
                            if len(r) == 1:
                                x1 = j
                            elif len(r) == 2:
                                x2 = j
                    if len(r) == 2 and in_s.count("x * * 2") == 1 and in_s.find("x * * 2") == 0:
                        print('Найдено приведенное квадратное уравнение: ' + in_s)
                        a, b, c = 1, -x1 - x2, x1 * x2
                        print(f'a = 1, b = {b}, c = {c}')
                    s = str(r)
                    if s != answer and answer != "":
                        cv2.rectangle(im, (i[0], i[1]), (i[2], i[3]), (255, 255, 0), 1)
                        cv2.putText(im, str(s), (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 0, 0), 2)
                    else:
                        if s != "" and answer == "":
                            answer = s
                        cv2.rectangle(im, (i[0], i[1]), (i[2], i[3]), (0, 255, 200), 1)
                        cv2.putText(im, str(s), (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), 2)
                except Exception as e:
                    # print(f'Не удалось решить уравнение: {s}')
                    if show_rows:
                        cv2.putText(im, str(s), (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 0), 2)

        ind_answer = s.find("Answer")
        if ind_answer > -1:
            s = s[ind_answer + 6:]
            if s[:2] == "01":
                s = "0," + s[2:]
                s = s.replace(", * *", ",")
                s = '[' + s + ']'
            print("Ответ:", s)
            s = str(s)
            if s != answer:
                cv2.rectangle(im, (i[0], i[1]), (i[2], i[3]), (255, 0, 0), 1)
            else:
                cv2.rectangle(im, (i[0], i[1]), (i[2], i[3]), (200, 255, 200), 1)

    st.image(im)