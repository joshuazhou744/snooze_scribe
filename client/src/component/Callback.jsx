import { useEffect } from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import { useNavigate } from 'react-router-dom';

const Callback = () => {
    const { isLoading, error } = useAuth0();
    const navigate = useNavigate();

    useEffect(() => {
        // Let Auth0Provider handle the callback automatically
        if (!isLoading && !error) {
            navigate('/');
        }
    }, [isLoading, error, navigate]);

    if (error) {
        return <div>Authentication Error: {error.message}</div>;
    }

    return <div>Loading...</div>;
};

export default Callback;