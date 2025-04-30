from django.db import models
from PIL import Image
import numpy as np
import insightface

# Carga el modelo una sola vez
_face_model = None
def get_face_model():
    global _face_model
    if _face_model is None:
        _face_model = insightface.app.FaceAnalysis(name='buffalo_l')
        _face_model.prepare(ctx_id=0, det_size=(640, 640))
    return _face_model

# 🧠 Aquí va el modelo Person
class Person(models.Model):
    dni = models.CharField(max_length=20, unique=True)
    nombre = models.CharField(max_length=100)
    apellidos = models.CharField(max_length=100)
    edad = models.IntegerField(null=True, blank=True)
    genero = models.CharField(max_length=10, null=True, blank=True)

    def __str__(self):
        return f"{self.nombre} {self.apellidos} ({self.dni})"

# 🧠 Aquí el modelo FaceImage
class FaceImage(models.Model):
    person = models.ForeignKey(Person, on_delete=models.CASCADE, related_name='images')
    imagen = models.ImageField(upload_to='uploads/')
    embedding = models.JSONField(null=True, blank=True)

    def save(self, *args, **kwargs):
        if not self.embedding and self.imagen:
            image = Image.open(self.imagen)
            image = image.convert('RGB')
            image = np.array(image)

            face_model = get_face_model()
            faces = face_model.get(image)

            if faces:
                self.embedding = faces[0].embedding.tolist()
            else:
                raise ValueError("No se detectó ninguna cara en la imagen subida.")

        super().save(*args, **kwargs)

    def __str__(self):
        return f"Imagen de {self.person.nombre} {self.person.apellidos}"
