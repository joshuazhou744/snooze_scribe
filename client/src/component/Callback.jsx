import React, { useEffect } from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import { useNavigate } from 'react-router-dom';

const Callback = () => {
    const {handleRedirectCallback, isAuthenticated} = useAuth0();
    const navigate = useNavigate();

    useEffect(() => {
    const processRedirect = async () => {
        try {
        await handleRedirectCallback();
        navigate('/');
        } catch (error) {
        console.error("Error handling redirect callback:", error);
        }
        }
    if (!isAuthenticated) {
        processRedirect();
      } else {
        navigate('/');
      }
    }, [handleRedirectCallback, navigate, isAuthenticated]);

    return <div>Loading...</div>;
};

export default Callback;