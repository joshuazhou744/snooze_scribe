import apiClient from './apiClient';

export async function fetchAudioFiles() {
  const { data } = await apiClient.get('/audio-files');
  return data;
}

export async function uploadAudioFile(blob, fileName) {
  const formData = new FormData();
  formData.append('file', blob, fileName);

  const { data } = await apiClient.post('/upload-audio', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return data;
}

export async function deleteAudioFile(fileId) {
  await apiClient.delete(`/audio-file/${fileId}`);
}

export async function deleteAllAudioFiles() {
  await apiClient.delete('/audio-files/all');
}

export async function classifyAudioFile(fileId) {
  const { data } = await apiClient.post(`/classify-audio/${fileId}`, {});
  return data;
}
