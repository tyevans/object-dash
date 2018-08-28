from django import forms

from objectdash.web.models import ExampleAnnotation, ExampleImage, PBObjectDetector


class UploadImageForm(forms.Form):
    image_file = forms.FileField(label="Image File:")


class ClassifyImageForm(UploadImageForm):
    min_confidence = forms.IntegerField(label="Min Confidence")


class VerificationReportForm(forms.Form):
    title = forms.TextInput()
    source = forms.ChoiceField(choices=[], required=False)
    labels = forms.MultipleChoiceField(
        choices=[],
        widget=forms.CheckboxSelectMultiple,
        required=False
    )
    models = forms.MultipleChoiceField(
        choices=[],
        widget=forms.CheckboxSelectMultiple,
        required=False
    )

    def __init__(self, *args, **kwargs):
        super(VerificationReportForm, self).__init__(*args, **kwargs)

        sources = ExampleImage.objects.all().values_list("source", "source").distinct()
        self.fields['source'].choices = list(set(sources))

        labels = ExampleAnnotation.objects.all().values_list("label", "label").distinct()
        self.fields['labels'].choices = labels

        models = PBObjectDetector.objects.all().values_list("name", "name").distinct()
        self.fields['models'].choices = models
