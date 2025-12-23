import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL;

const apiClient = axios.create({
  baseURL: API_BASE_URL,
});

let authHandlers = {
  getToken: null,
  onAuthFailure: null,
};

export const setAuthHandlers = ({ getToken, onAuthFailure }) => {
  authHandlers = { getToken, onAuthFailure };
};

apiClient.interceptors.request.use(
  async (config) => {
    if (!authHandlers.getToken) {
      return config;
    }
    try {
      const token = await authHandlers.getToken();
      if (token) {
        config.headers = {
          ...config.headers,
          Authorization: `Bearer ${token}`,
        };
      }
      return config;
    } catch (error) {
      if (authHandlers.onAuthFailure) {
        authHandlers.onAuthFailure(error);
      }
      return Promise.reject(error);
    }
  },
  (error) => Promise.reject(error)
);

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (authHandlers.onAuthFailure) {
      authHandlers.onAuthFailure(error);
    }
    return Promise.reject(error);
  }
);

export default apiClient;
