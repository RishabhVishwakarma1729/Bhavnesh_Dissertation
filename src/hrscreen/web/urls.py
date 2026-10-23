from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("screen/", views.screen, name="screen"),
    path("runs/", views.run_list, name="run_list"),
    path("runs/<int:run_id>/", views.run_detail, name="run_detail"),
    path("runs/<int:run_id>/ranking.csv", views.run_csv, name="run_csv"),
    path("runs/<int:run_id>/delete/", views.run_delete, name="run_delete"),
    path("benchmark/", views.benchmark, name="benchmark"),
    path("fairness/", views.fairness, name="fairness"),
    path("survey/", views.survey, name="survey"),
    path("survey/<int:run_id>/", views.survey_detail, name="survey_detail"),
    path("survey/<int:run_id>/figure/<str:name>.png", views.survey_figure, name="survey_figure"),
    path("survey/<int:run_id>/download/<str:kind>/", views.survey_file, name="survey_file"),
    path("survey/<int:run_id>/delete/", views.survey_delete, name="survey_delete"),
    path("questionnaire/", views.questionnaire, name="questionnaire"),
    path("questionnaire/download/<str:fmt>/", views.questionnaire_download, name="questionnaire_download"),
    path("about/", views.about, name="about"),
]
