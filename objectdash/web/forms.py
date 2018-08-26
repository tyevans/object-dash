from django import forms

class UploadImageForm(forms.Form):
    image_file = forms.FileField(label="Image File:")


class ClassifyImageForm(UploadImageForm):
    min_confidence = forms.IntegerField(label="Min Confidence")