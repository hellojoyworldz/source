# 강제로 회원가입
from sqlalchemy.orm import Session
from backend.repository.db_init import SessionLocal
from backend.repository.models import Customer

db = SessionLocal()

DEFAULT_CUSTOMERS = [
    Customer(name="홍길동", phone="010-1111-1111"),
    Customer(name="김철수", phone="010-2222-2222"),
    Customer(name="김영희", phone="010-3333-3333"),
]


def seed_customers(db: Session):
    """
    customer 테이블에 기본 회원 테이터 삽입
    (중복 실행 방지: 이미 데이터가 있으면 건너뜀)
    """

    existing = db.query(Customer).first()
    if existing:
        print("[Seed] customer 테이블에 이미 데이터가 있습니다.")
        return

    # 하나 실패 다 롤백
    db.add_all(DEFAULT_CUSTOMERS)
    db.commit()
    db.close()
    print(f"[Seed] 기본 회원 {len(DEFAULT_CUSTOMERS)} 명 삽입 완료")
