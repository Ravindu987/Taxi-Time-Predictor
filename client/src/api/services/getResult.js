import axiosClient from '../axios';

const getResult = async (data) => {
    try {
        const response = await axiosClient.post('', data);
        return response.data;
    } catch (error) {
        console.log(error);
    }
}

export default getResult;