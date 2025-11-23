import { Auth0Provider } from "@auth0/auth0-react";
import {useNavigate} from "react-router-dom"
import PropTypes from 'prop-types';

export default function Auth0ProviderWithHistory({ children = null }) {
  
    const navigate = useNavigate();

    const domain = import.meta.env.VITE_AUTH0_DOMAIN;
    const clientId = import.meta.env.VITE_AUTH0_CLIENT_ID;
    const audience = import.meta.env.VITE_AUTH0_AUDIENCE;


    const onRedirectCallback = (appState) => {
        navigate(appState?.returnTo || window.location.pathname)
    }

    return (
        <div>
            <Auth0Provider
                domain={domain}
                clientId={clientId}
                authorizationParams={{
                    audience: audience,
                    redirect_uri: `${window.location.origin}/callback`,
                    scope: "openid profile email offline_access"
                }}
                onRedirectCallback={onRedirectCallback}
                useRefreshTokens={true}
                cacheLocation="localstorage"
                >
                {children}
            </Auth0Provider>
        </div>
  )
}

Auth0ProviderWithHistory.propTypes = {
  children: PropTypes.node,
};
