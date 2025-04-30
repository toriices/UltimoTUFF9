from django.db import models
from PIL import Image
import numpy as np
import face_recognition
import tempfile

# 🧠 Modelo Person
class Person(models.Model):
    dni = models.CharField(max_length=20, unique=True)
    nombre = models.CharField(max_length=100)
    apellidos = models.CharField(max_length=100)
    edad = models.IntegerField(null=True, blank=True)
    genero = models.CharField(max_length=10, null=True, blank=True)

    def __str__(self):
        return f"{self.nombre} {self.apellidos} ({self.dni})"

# 🧠 Modelo FaceImage
class FaceImage(models.Model):
    person = models.ForeignKey(Person, on_delete=models.CASCADE, related_name='images')
    imagen = models.ImageField(upload_to='uploads/')
    embedding = models.JSONField(null=True, blank=True)

    def save(self, *args, **kwargs):
        if not self.embedding and self.imagen:
            image = Image.open(self.imagen)
            image = image.convert('RGB')
            image = image.resize((640, 640))  # mejora detección

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp.name)
                image_array = face_recognition.load_image_file(tmp.name)

            face_locations = face_recognition.face_locations(image_array, model='cnn')
            if face_locations:
                encodings = face_recognition.face_encodings(image_array, face_locations)
                if encodings:
                    self.embedding = encodings[0].tolist()
                else:
                    raise ValueError("No se pudo calcular el embedding de la cara.")
            else:
                raise ValueError("No se detectó ninguna cara en la imagen subida.")

        super().save(*args, **kwargs)

    def __str__(self):
        return f"Imagen de {self.person.nombre} {self.person.apellidos}"
