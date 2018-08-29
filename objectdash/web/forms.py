from django import forms

from objectdash.web.models import Annotation, AnnotatedImage, ObjectDetector


class UploadImageForm(forms.Form):
    image_file = forms.FileField(label="Image File:")


class ClassifyImageForm(UploadImageForm):
    min_confidence = forms.IntegerField(label="Min Confidence")


class AnnotationFilterForm(forms.Form):
    title = forms.TextInput()
    source = forms.ChoiceField(choices=[], required=False)
    labels = forms.MultipleChoiceField(
        choices=[],
        widget=forms.CheckboxSelectMultiple,
        required=False
    )

    def __init__(self, *args, **kwargs):
        super(AnnotationFilterForm, self).__init__(*args, **kwargs)

        sources = AnnotatedImage.objects.all().values_list("source", "source").distinct()
        self.fields['source'].choices = list(set(sources))

        labels = Annotation.objects.all().values_list("label", "label").distinct()
        self.fields['labels'].choices = labels
