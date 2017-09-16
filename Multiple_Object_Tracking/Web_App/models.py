from django.db import models
from django.contrib.auth.models import User


class Movie(models.Model):
    path = models.CharField(max_length=128, default="File_path")
    file = models.FileField(upload_to='files/', default="Server_path")
    creation_date = models.DateTimeField(auto_now=True)
    my_user = models.ForeignKey(User)

    @property
    def movie_info(self):
        return "Added: {} by {}".format(self.creation_date
                                            .strftime("%Y-%m-%d, %H:%M:%S"),
                                        self.my_user.username)

    def __str__(self):
        return "ID:" + str(self.id) + " " + self.movie_info


class Detection(models.Model):
    user = models.ForeignKey(User)
    movie = models.ForeignKey(Movie)
    data = models.TextField()
    creation_date = models.DateTimeField(auto_now=True)


class Estimate(models.Model):
    user = models.ForeignKey(User)
    movie = models.ForeignKey(Movie)
    data = models.TextField()
    creation_date = models.DateTimeField(auto_now=True)

