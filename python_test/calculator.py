# 계산 기능 클래스 import
from calculating import Operations

# 계산기 클래스
class Calculator:
  # 연산자 전 초기값 필요
  def __init__(self, firstValue):
    self.firstValue = firstValue
  def insertValue(self):
    Operations.result = self.firstValue
    print(Operations.result)
    while True:
      isOperation = input("연산자 : ")
      if isOperation != "=":
        operating = Operations(isOperation, int(input("값 : ")))
        operating.calculating()
      else:
        print(f"결과값 : {Operations.result}")
        Operations.result = 0
        break

calculatorImplement = Calculator(int(input("값 : ")))
calculatorImplement.insertValue()
