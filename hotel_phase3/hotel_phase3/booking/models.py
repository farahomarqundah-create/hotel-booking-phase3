from django.db import models

class HotelPrediction(models.Model):
    lead_time = models.IntegerField()
    adults = models.IntegerField()
    stays_in_week_nights = models.IntegerField()
    prediction = models.IntegerField()
    
    def __str__(self):
        
        return str(self.prediction)
 