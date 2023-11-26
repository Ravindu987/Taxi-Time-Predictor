import axios from 'axios';

const axiosClient = axios.create({
    baseURL: 'http://localhost:3000/predict',
    timeout: 1000,
    accessControlAllowOrigin: '*',
    accessControlAllowHeaders: '*',
});

export default axiosClient;