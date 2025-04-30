from django.contrib import admin
from apps.recognition.models import Person, FaceImage

class FaceImageInline(admin.TabularInline):
    model = FaceImage
    extra = 1  # Cuántos campos vacíos aparecen para añadir imágenes nuevas

class PersonAdmin(admin.ModelAdmin):
    list_display = ('dni', 'nombre', 'apellidos')
    search_fields = ('dni', 'nombre', 'apellidos')
    inlines = [FaceImageInline]  # Mostrar imágenes relacionadas dentro del admin de Person

admin.site.register(Person, PersonAdmin)
admin.site.register(FaceImage)
