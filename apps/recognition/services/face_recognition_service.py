import numpy as np
import insightface
from apps.recognition.models.person import Person, FaceImage
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

class FaceRecognitionService:
    def __init__(self):
        self.model = insightface.app.FaceAnalysis(name='buffalo_l')
        self.model.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding(self, image_file):
        img = Image.open(image_file)
        img = img.convert('RGB')  # Asegurar 3 canales
        img = np.array(img)

        faces = self.model.get(img)

        if len(faces) == 0:
            return None

        face = faces[0]
        embedding = face.embedding.tolist()
        edad = int(face.age)
        genero = 'Male' if face.gender == 1 else 'Female'

        return embedding, edad, genero

    def find_best_match(self, embedding, threshold=0.4):
        all_images = FaceImage.objects.select_related('person').all()
        best_match = None
        best_score = -1

        if embedding is None:
            return None

        for face_image in all_images:
            if face_image.embedding:
                stored_embedding = np.array(face_image.embedding)
                sim = cosine_similarity([embedding], [stored_embedding])[0][0]

                if sim > best_score and sim >= threshold:
                    best_score = sim
                    best_match = face_image.person

        return best_match
