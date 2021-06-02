from data.electricity.preprocess_elect import elect_loader
from data.walmart.preprocess_walmart import walmart_loader
from data.ohio.preprocessing_ohio import preprocess_ohio_point_forecast
from data.rossmann.rossmann_preprocessing import generate_rossmann_data

dataset_loader = dict(
    electricity = elect_loader,
    rossmann = generate_rossmann_data,
    walmart = walmart_loader,
    ohio = preprocess_ohio_point_forecast,
)