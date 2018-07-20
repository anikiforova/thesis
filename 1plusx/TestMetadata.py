


class TestMetadata:
	impression_count 	= 1.0
	click_count 		= 0.0
	local_clicks 		= 0.0
	local_count 		= 1.0
	total_local_count 	= 0.0
	missed_clicks 		= 0.0
	total_clicks 		= 0.0
	local_missed_clicks = 0.0
	total_local_clicks 	= 0.0
	impressions_per_recommendation_group = 0.0

	SE = 0.0
	local_SE = 0.0
	cumulative_SE = 0.0 

	warmup = True

	users_to_update = list()
	clicks_to_update = list()

	def update_recommendations():

		users_to_update = list()
		clicks_to_update = list()

		self.SE = 0.0
		self.impressions_per_recommendation_group = 0.0

		total_local_count = 0.0
		warmup = False	