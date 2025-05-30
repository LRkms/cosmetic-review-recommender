import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QStringListModel
from recommend_app_ui import Ui_Dialog  # pyuic5로 생성된 recommend_app.ui 파이썬 모듈
from test import recommend

class RecommendDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # 1) QLabel 자체 크기를 810×50으로 고정
        self.lbl_banner.setFixedSize(810, 50)
        self.lbl_item.setFixedSize(200, 230)

        # 2) 배너 이미지를 로드
        banner = QtGui.QPixmap("logo.png")
        if banner.isNull():
            raise FileNotFoundError("⚠️ logo.png 파일을 찾을 수 없습니다.")
        cosmetic = QtGui.QPixmap("cosmetic.png")
        if cosmetic.isNull():
            raise FileNotFoundError("⚠️ cosmetic.png 파일을 찾을 수 없습니다.")

        # 3) 런타임에 정확히 810×50으로 부드럽게 스케일
        banner_scaled = banner.scaled(
            810, 50,
            QtCore.Qt.IgnoreAspectRatio,  # 비율 무시하고 딱 맞춤
            QtCore.Qt.SmoothTransformation  # 고품질 리샘플링
        )
        cosmetic_scaled = cosmetic.scaled(
            200, 230,
            QtCore.Qt.IgnoreAspectRatio,  # 비율 무시하고 딱 맞춤
            QtCore.Qt.SmoothTransformation  # 고품질 리샘플링
        )

        # 4) QLabel에 세팅
        self.lbl_banner.setPixmap(banner_scaled)
        self.lbl_banner.setScaledContents(False)

        self.lbl_item.setPixmap(cosmetic_scaled)
        self.lbl_item.setScaledContents(False)

        # 콤보박스 기본 항목 저장
        self.default_skins = [self.cb_skin.itemText(i) for i in range(self.cb_skin.count())]
        self.default_cats  = [self.cb_category.itemText(i) for i in range(self.cb_category.count())]

        # 성별 매핑 설정
        self.skin_map = {
            "male":   ["건성", "지성", "복합성", "모름"],
            "female": ["건성", "지성", "복합성", "모름"]
        }
        self.cat_map = {
            "male":   ["스킨케어", "쉐이빙/왁싱"],
            "female": ["스킨케어", "메이크업", "선케어"]
        }

        # 신호 연결
        self.check_male.toggled.connect(lambda checked: self.on_gender("male", checked))
        self.check_female.toggled.connect(lambda checked: self.on_gender("female", checked))
        self.line_cmd.returnPressed.connect(self.on_search)
        # OK 버튼 클릭 시 검색
        self.buttonBox.accepted.disconnect()  # 기존 accept 연결 해제
        self.buttonBox.accepted.connect(self.on_search)

    def on_gender(self, gender: str, checked: bool):
        other = "female" if gender == "male" else "male"
        if checked:
            getattr(self, f"check_{other}").setChecked(False)
            self.cb_skin.clear()
            self.cb_skin.addItems(self.skin_map[gender])
            self.cb_category.clear()
            self.cb_category.addItems(self.cat_map[gender])
        else:
            self.cb_skin.clear()
            self.cb_skin.addItems(self.default_skins)
            self.cb_category.clear()
            self.cb_category.addItems(self.default_cats)

    def on_search(self):
        # 선택된 성별
        genders = []
        if self.check_male.isChecked():
            genders.append("남성")
        if self.check_female.isChecked():
            genders.append("여성")
        gender_str = genders[0] if genders else ""

        # 콤보박스 항목
        skin = self.cb_skin.currentText()
        cat = self.cb_category.currentText()
        custom = self.line_cmd.text().strip()

        # 키워드 조합 (플레이스홀더 필터링)
        parts = [gender_str, skin, cat, custom]
        keywords = " ".join([p for p in parts if p and p not in ["피부타입", "카테고리"]])

        # 추천 호출
        results = recommend(keywords, top_n=10)

        # ListView 모델 설정
        model = QStringListModel()
        model.setStringList(results.tolist())
        self.list_table.setModel(model)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    dlg = RecommendDialog()
    dlg.show()
    sys.exit(app.exec_())
