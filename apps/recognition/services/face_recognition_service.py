import numpy as np
import insightface
from apps.recognition.models.person import Person, FaceImage
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, UnidentifiedImageError

class FaceRecognitionService:
    def __init__(self):
        self.model = insightface.app.FaceAnalysis(name='buffalo_l')
        self.model.prepare(ctx_id=-1, det_size=(640, 640))  # CPU compatible

    def get_embedding(self, image_file):
        try:
            img = Image.open(image_file)
            img = img.convert('RGB')
            img = np.array(img)
        except UnidentifiedImageError:
            return None
        except Exception:
            return None

        faces = self.model.get(img)

        if not faces:
            return None

        face = faces[0]
        embedding = face.embedding.tolist()
        edad = int(face.age)
        genero = 'Male' if face.gender == 1 else 'Female'

        return embedding, edad, genero

    def find_best_match(self, embedding, threshold=0.4):
        if embedding is None:
            return None

        all_images = FaceImage.objects.select_related('person').all()
        best_match = None
        best_score = -1

        for face_image in all_images:
            if face_image.embedding:
                stored_embedding = np.array(face_image.embedding)
                if stored_embedding.shape != (512,):
                    continue

                sim = cosine_similarity([embedding], [stored_embedding])[0][0]

                if sim > best_score and sim >= threshold:
                    best_score = sim
                    best_match = face_image.person

        return best_match
