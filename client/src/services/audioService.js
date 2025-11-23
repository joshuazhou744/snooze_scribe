import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL;

const authHeaders = (token) => ({
  Authorization: `Bearer ${token}`,
});

export async function fetchAudioFiles(token) {
  const { data } = await axios.get(`${API_BASE_URL}/audio-files`, {
    headers: authHeaders(token),
  });
  return data;
}

export async function uploadAudioFile(token, blob, fileName) {
  const formData = new FormData();
  formData.append('file', blob, fileName);

  const { data } = await axios.post(`${API_BASE_URL}/upload-audio`, formData, {
    headers: {
      ...authHeaders(token),
      'Content-Type': 'multipart/form-data',
    },
  });
  return data;
}

export async function deleteAudioFile(token, fileId) {
  await axios.delete(`${API_BASE_URL}/audio-file/${fileId}`, {
    headers: authHeaders(token),
  });
}

export async function deleteAllAudioFiles(token) {
  await axios.delete(`${API_BASE_URL}/audio-files/all`, {
    headers: authHeaders(token),
  });
}

export async function classifyAudioFile(token, fileId) {
  const { data } = await axios.post(
    `${API_BASE_URL}/classify-audio/${fileId}`,
    {},
    {
      headers: authHeaders(token),
    }
  );
  return data;
}
