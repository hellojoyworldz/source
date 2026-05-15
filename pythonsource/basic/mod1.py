# 모듈 생성
def add(a,b):
    return a+b

def sub(a,b):
    return a-b
 


# 모듈 안 기능들이 잘 만들어졌는지 확인
# mod1이 직접 실행될 때만 실행. 외부에서 실행 될 때는 실행 안 됨
if __name__ == "__main__":
    print(add(3,4))
    print(sub(3,4))
