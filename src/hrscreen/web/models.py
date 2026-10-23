"""Database tables. Resume text is deliberately NOT stored: a run keeps only each candidate's label
(the file name), score and parsed attributes, so deleting a run leaves nothing sensitive behind."""

from django.db import models


class ScreeningRun(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    job_title = models.CharField(max_length=200)
    jd_text = models.TextField()
    model = models.CharField(max_length=20)
    is_demo = models.BooleanField(default=False, help_text="Candidates were SYNTHETIC demo resumes")

    class Meta:
        ordering = ["-created_at", "-id"]

    MODEL_LABELS = {"tfidf": "TF-IDF similarity", "rules": "Keyword rules", "embeddings": "Sentence embeddings"}

    def __str__(self) -> str:
        return f"{self.job_title} ({self.model}) {self.created_at:%Y-%m-%d %H:%M}"

    def get_model_label(self) -> str:
        return self.MODEL_LABELS.get(self.model, self.model)

    @property
    def ranked(self):
        return self.candidates.filter(error="").order_by("rank")

    @property
    def unreadable(self):
        return self.candidates.exclude(error="")


class Candidate(models.Model):
    run = models.ForeignKey(ScreeningRun, related_name="candidates", on_delete=models.CASCADE)
    label = models.CharField(max_length=255)
    rank = models.PositiveIntegerField(null=True, blank=True)
    score = models.FloatField(null=True, blank=True)
    matched_skills = models.JSONField(default=list)
    missing_skills = models.JSONField(default=list)
    years_experience = models.FloatField(null=True, blank=True)
    education_level = models.CharField(max_length=20, blank=True)
    n_words = models.PositiveIntegerField(default=0)
    error = models.CharField(max_length=255, blank=True, help_text="Why the file could not be ranked")

    class Meta:
        ordering = ["rank", "id"]


class SurveyRun(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    source_name = models.CharField(max_length=255)
    n_responses = models.PositiveIntegerField()
    is_synthetic = models.BooleanField(default=False)
    report_md = models.TextField()
    results = models.JSONField(default=dict, help_text="Tables rendered by the survey page")

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self) -> str:
        return f"{self.source_name} (n={self.n_responses})"
