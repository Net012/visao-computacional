import cv2
import mediapipe as mp
import time

class DetectorDeMaos:
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackingConfidence=0.5):
        self.modo = mode
        self.maximoMaos = maxHands
        self.confiancaDeteccao = detectionConfidence
        self.confiancaRastreamento = trackingConfidence

        self.moduloMaos = mp.solutions.hands
        self.maos = self.moduloMaos.Hands(self.modo, self.maximoMaos,
                                          min_detection_confidence=self.confiancaDeteccao,
                                          min_tracking_confidence=self.confiancaRastreamento)
        self.drawer = mp.solutions.drawing_utils

    def encontrarMaos(self, frame, draw=True):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.resultados = self.maos.process(frame_rgb)

        if self.resultados.multi_hand_landmarks:
            for marcacaoMao in self.resultados.multi_hand_landmarks:
                if draw:
                    self.drawer.draw_landmarks(frame, marcacaoMao,
                                               self.moduloMaos.HAND_CONNECTIONS)
        return frame

    def encontrarPosicoes(self, frame, numeroMao=0, draw=True):
        listaMarcacoes = []
        if self.resultados.multi_hand_landmarks:
            maoAtual = self.resultados.multi_hand_landmarks[numeroMao]
            for id_ponto, ponto in enumerate(maoAtual.landmark):
                altura, largura, canais = frame.shape
                pos_x, pos_y = int(ponto.x * largura), int(ponto.y * altura)
                listaMarcacoes.append([id_ponto, pos_x, pos_y])
                if draw:
                    cv2.circle(frame, (pos_x, pos_y), 15, (255, 0, 255), cv2.FILLED)

        return listaMarcacoes

def main():
    prevTime = 0
    time_now = 0
    camera = cv2.VideoCapture(0)
    detector = DetectorDeMaos()

    while True:
        success, frame = camera.read()
        if not success:
            print("Erro ao capturar a imagem da câmera.")
            break

        frame = detector.encontrarMaos(frame)
        listaMarcacoes = detector.encontrarPosicoes(frame)
        if len(listaMarcacoes) != 0:
            print(listaMarcacoes[4])

        time_now = time.time()
        fps = 1 / (time_now - prevTime)
        prevTime = time_now

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Imagem", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
