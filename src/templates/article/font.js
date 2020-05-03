import React from 'react';
import Types from 'prop-types';
import { Helmet } from 'react-helmet';

const Font = ({ id, fontPath }) => {
  return (
    <Helmet>
      <link rel="preload" href={fontPath} as="font" crossOrigin="anonymous" />
      <link
        rel="preload"
        href="/font/fira_code.woff2"
        as="font"
        crossOrigin="anonymous"
      />
      <style>
        {`
            @font-face {
              font-family: ${id}_font;
              src: url('${fontPath}');
            }
            @font-face {
              font-family: fira_code;
              src: url('/font/fira_code.woff2');
            }
          `}
      </style>
    </Helmet>
  );
};
Font.propTypes = {
  id: Types.string.isRequired,
  fontPath: Types.string.isRequired,
};

export default Font;
