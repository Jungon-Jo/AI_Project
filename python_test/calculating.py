# 계산 작업 클래스
class Operations:
  # 결과값에 대한 변수 선언
  result = 0
  # 생성자
  def __init__(self, operator, nums):
    # 변수 설정(값과 연산자를 입력)
    self.nums = nums
    self.operator = operator
  # 연산 메서드
  def calculating(self):
      if self.operator == "+":
        print(self.result)
        self.result += self.nums
        print(f"calculating : {self.result}")
      elif self.operator == "-":
        print(self.result)
        self.result -= self.nums
        print(f"calculating : {self.result}")
      elif self.operator == "*":
        print(self.result)
        self.result *= self.nums
        print(f"calculating : {self.result}")
      elif self.operator == "/":
        print(self.result)
        self.result /= self.nums
        print(f"calculating : {self.result}")
 





    