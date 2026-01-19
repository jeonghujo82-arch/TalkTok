from app import create_app, db # app.py에서 create_app 함수를 임포트합니다.

app = create_app() # create_app 함수를 호출하여 Flask 앱 인스턴스를 생성합니다.

if __name__ == '__main__':
    with app.app_context():
        db.create_all() # 앱 컨텍스트 내에서 데이터베이스 테이블을 생성합니다.
    app.run(debug=True)