import React, { useEffect, useState } from 'react';
import { WiHumidity, WiBarometer, WiStrongWind, WiThermometer, WiSunrise, WiSunset, WiCloudyGusts } from 'react-icons/wi';
import { Search } from 'lucide-react';

// API key (Note: In a production environment, this should be stored securely)
const API_KEY = '3c39dd8447beb9c1548b28b78b602a26';

// Utility functions
const fetchData = async (URL) => {
  const response = await fetch(`${URL}&appid=${API_KEY}`);
  if (!response.ok) {
    console.error(`Error ${response.status}: ${response.statusText}`);
    throw new Error('Network response was not ok');
  }
  return response.json();
};

// OpenWeatherAPI Endpoints
const url = {
  currentWeather: (lat, lon) =>
    `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}`,
  airPollution: (lat, lon) =>
    `https://api.openweathermap.org/data/2.5/air_pollution?lat=${lat}&lon=${lon}`,
  forecast: (lat, lon) =>
    `https://api.openweathermap.org/data/2.5/forecast?lat=${lat}&lon=${lon}`,
  geocoding: (query) => `https://api.openweathermap.org/geo/1.0/direct?q=${query}&limit=5`,
  disasters: (lat, lon) => `https://api.openweathermap.org/data/2.5/onecall?lat=${lat}&lon=${lon}`
};

// Helper functions
const getDate = (dateUnix, timezone) => {
  const date = new Date((dateUnix + timezone) * 1000);
  return date.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' });
};

const getTime = (timeUnix, timezone) => {
  const date = new Date((timeUnix + timezone) * 1000);
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

export default function Climate() {
  const [loading, setLoading] = useState(false);
  const [weatherData, setWeatherData] = useState(null);
  const [forecastData, setForecastData] = useState([]);
  const [currentWeather, setCurrentWeather] = useState([]);
  const [airQualityIndex, setAirQualityIndex] = useState(null);
  const [error, setError] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedLocation, setSelectedLocation] = useState('');
  const [disasterAlerts, setDisasterAlerts] = useState(null);
  const [suggestions, setSuggestions] = useState([]);  // For storing location suggestions
  const [debounceTimeout, setDebounceTimeout] = useState(null);  // For debounce control

  // Fetch weather data
  const fetchWeatherData = async (lat, lon) => {
    setLoading(true);
    setError(false);
    try {
      const currentWeatherData = await fetchData(url.currentWeather(lat, lon));
      const airPollutionData = await fetchData(url.airPollution(lat, lon));
      const forecast = await fetchData(url.forecast(lat, lon));
      const disasterData = await fetchData(url.disasters(lat, lon));

      setWeatherData(currentWeatherData);
      setAirQualityIndex(airPollutionData.list[0].main.aqi);
      setDisasterAlerts(disasterData?.alerts || null);

      const now = new Date();
      const cutoffTime = new Date(now.getTime() + 24 * 60 * 60 * 1000);

      const next24HoursForecast = forecast.list.filter((entry) => {
        const entryDate = new Date(entry.dt * 1000);
        return entryDate >= now && entryDate <= cutoffTime;
      });

      setCurrentWeather(next24HoursForecast);
      setForecastData(forecast.list.filter((_, index) => index % 8 === 0));
    } catch (err) {
      console.error(err);
      setError(true);
    } finally {
      setLoading(false);
    }
  };

  // Handle search term change with debounce
  const handleSearchChange = async (e) => {
    const value = e.target.value;
    setSearchTerm(value);

    if (debounceTimeout) {
      clearTimeout(debounceTimeout);
    }

    if (value.length === 0) {
      setSuggestions([]);
      return;
    }

    const newDebounceTimeout = setTimeout(async () => {
      try {
        const locations = await fetchData(url.geocoding(value));
        setSuggestions(locations);  // Set the location suggestions
      } catch (err) {
        console.error('Error fetching geocoding data:', err);
      }
    }, 500);  // 500ms debounce time

    setDebounceTimeout(newDebounceTimeout);
  };

  // Handle suggestion click
  const handleSuggestionClick = (location) => {
    const { lat, lon, name, state, country } = location;
    setSelectedLocation(`${name}${state ? ', ' + state : ''}, ${country}`);
    setSearchTerm(`${name}${state ? ', ' + state : ''}, ${country}`);
    setSuggestions([]);
    fetchWeatherData(lat, lon);
  };

  // Handle current location
  const handleCurrentLocation = () => {
    if (!navigator.geolocation) {
      alert('Geolocation is not supported by your browser.');
      return;
    }

    setLoading(true);
    setError(false);
    setWeatherData(null);
    setForecastData([]);
    setCurrentWeather([]);
    setAirQualityIndex(null);

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        setSelectedLocation('Your Current Location');
        fetchWeatherData(latitude, longitude);
      },
      (err) => {
        console.error('Error fetching current location:', err);
        setError(true);
        setLoading(false);
        alert('Unable to retrieve your location.');
      }
    );
  };

  return (
    <div className="min-h-screen flex flex-col items-center bg-gradient-to-br from-blue-100 to-green-100 p-5 w-full mt-14">
      <div className="w-full max-w-4xl mb-1 mt-24 relative">
        <form onSubmit={(e) => e.preventDefault()} className="flex w-full mb-4">
          <input
            type="text"
            value={searchTerm}
            onChange={handleSearchChange}
            className="p-4 border border-gray-300 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 w-full transition duration-300 ease-in-out"
            placeholder="Search for a location (City, State, Country)"
          />
          <button
            type="button"
            onClick={() => handleSearchChange({ target: { value: searchTerm } })}
            className="p-4 bg-blue-500 text-white rounded-r-lg hover:bg-blue-600 transition duration-300 ease-in-out"
          >
            <Search className="w-6 h-6" />
          </button>
        </form>

        {/* Display suggestions */}
        {suggestions.length > 0 && (
          <div className="absolute bg-white shadow-lg rounded-lg max-w-xl w-full mt-2 z-10">
            {suggestions.map((location, index) => (
              <div
                key={index}
                className="p-3 cursor-pointer hover:bg-gray-200"
                onClick={() => handleSuggestionClick(location)}
              >
                <p>{`${location.name}${location.state ? ', ' + location.state : ''}, ${location.country}`}</p>
              </div>
            ))}
          </div>
        )}

        <button
          onClick={handleCurrentLocation}
          className="w-full p-4 bg-green-500 text-white rounded-lg hover:bg-green-600 transition duration-300 ease-in-out flex items-center justify-center mt-4"
        >
          Use Current Location
        </button>
      </div>

      {loading && <p className="text-blue-800">Loading...</p>}
      {error && <p className="text-red-500">Error fetching data. Please try again.</p>}

      {/* Weather details and other components as before */}
      {weatherData && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full max-w-4xl">
          {/* Current Weather */}
          <div className="bg-white rounded-xl shadow-lg p-6 transition duration-300 hover:shadow-xl">
            <h2 className="text-3xl font-bold text-center text-gray-800 mb-4">
              {selectedLocation || weatherData.name}
            </h2>
            <div className="flex flex-col items-center">
              <img
                src={`https://openweathermap.org/img/wn/${weatherData.weather[0].icon}@4x.png`}
                alt={weatherData.weather[0].description}
                className="w-32 h-32 mb-4"
              />
              <p className="text-6xl font-semibold text-gray-800 mb-2">
                {Math.round(weatherData.main.temp - 273.15)}°C
              </p>
              <p className="text-xl text-gray-600 capitalize">{weatherData.weather[0].description}</p>
            </div>
          </div>

          {/* Weather Details */}
          <div className="bg-white rounded-xl shadow-lg p-6 transition duration-300 hover:shadow-xl">
            <h3 className="text-2xl font-semibold mb-4 text-center text-gray-800">Current Conditions</h3>
            <div className="grid grid-cols-2 gap-4">
              <WeatherDetail icon={WiHumidity} label="Humidity" value={`${weatherData.main.humidity}%`} />
              <WeatherDetail icon={WiThermometer} label="Feels Like" value={`${Math.round(weatherData.main.feels_like - 273.15)}°C`} />
              <WeatherDetail icon={WiStrongWind} label="Wind Speed" value={`${Math.round(weatherData.wind.speed * 3.6)} km/h`} />
              <WeatherDetail icon={WiBarometer} label="Pressure" value={`${weatherData.main.pressure} hPa`} />
              <WeatherDetail icon={WiSunrise} label="Sunrise" value={getTime(weatherData.sys.sunrise, weatherData.timezone)} />
              <WeatherDetail icon={WiSunset} label="Sunset" value={getTime(weatherData.sys.sunset, weatherData.timezone)} />
              <WeatherDetail icon={WiCloudyGusts} label="Air Quality" value={airQualityIndex ? `${airQualityIndex}` : 'N/A'} />
            </div>
          </div>
        </div>
      )}

      {/* 5-Day Forecast */}
      {forecastData.length > 0 && (
        <div className="mt-8 w-full max-w-4xl bg-white rounded-xl shadow-lg p-6 transition duration-300 hover:shadow-xl">
          <h3 className="text-2xl font-semibold mb-4 text-center text-gray-800">5-Day Forecast</h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {forecastData.map((forecast, index) => (
              <div key={index} className="flex flex-col items-center p-2 border rounded-lg">
                <p className="font-semibold text-gray-700">{getDate(forecast.dt, weatherData.timezone)}</p>
                <img
                  src={`https://openweathermap.org/img/wn/${forecast.weather[0].icon}@2x.png`}
                  alt={forecast.weather[0].description}
                  className="w-16 h-16 my-2"
                />
                <p className="text-lg font-semibold text-gray-800">{Math.round(forecast.main.temp - 273.15)}°C</p>
                <p className="text-sm text-gray-600 capitalize">{forecast.weather[0].description}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Hourly Forecast */}
      {currentWeather.length > 0 && (
        <div className="mt-8 w-full max-w-4xl bg-white rounded-xl shadow-lg p-6 transition duration-300 hover:shadow-xl">
          <h3 className="text-2xl font-semibold mb-4 text-center text-gray-800">Hourly Forecast</h3>
          <div className="overflow-x-auto">
            <div className="inline-flex space-x-4 pb-4">
              {currentWeather.map((entry, index) => (
                <div key={index} className="flex flex-col items-center p-2 border rounded-lg min-w-[100px]">
                  <p className="font-semibold text-gray-700">{getTime(entry.dt, weatherData.timezone)}</p>
                  <img
                    src={`https://openweathermap.org/img/wn/${entry.weather[0].icon}@2x.png`}
                    alt={entry.weather[0].description}
                    className="w-16 h-16 my-2"
                  />
                  <p className="text-lg font-semibold text-gray-800">{Math.round(entry.main.temp - 273.15)}°C</p>
                  <p className="text-sm text-gray-600 capitalize">{entry.weather[0].description}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Disaster Alerts */}
      {disasterAlerts && disasterAlerts.length > 0 ? (
        <div className="mt-8 w-full max-w-4xl bg-red-100 rounded-xl shadow-lg p-6 transition duration-300 hover:shadow-xl">
          <h3 className="text-2xl font-semibold mb-4 text-center text-red-800">Disaster Alerts</h3>
          <div className="space-y-4">
            {disasterAlerts.map((alert, index) => (
              <div key={index} className="p-4 bg-red-200 rounded-lg">
                <p className="font-semibold text-lg text-red-800">{alert.title}</p>
                <p className="text-sm text-red-700">{alert.description}</p>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="mt-8 w-full max-w-4xl bg-green-100 rounded-xl shadow-lg p-6 transition duration-300 hover:shadow-xl">
          <h3 className="text-2xl font-semibold mb-4 text-center text-green-800">No Threats for Now</h3>
          <p className="text-center text-green-700">Currently, there are no disaster alerts for this location.</p>
        </div>
      )}
    </div>
  );
}

function WeatherDetail({ icon: Icon, label, value }) {
  return (
    <div className="flex items-center space-x-2">
      <Icon className="w-8 h-8 text-blue-500" />
      <div>
        <p className="text-sm text-gray-600">{label}</p>
        <p className="text-lg font-semibold text-gray-800">{value}</p>
      </div>
    </div>
  );
}
