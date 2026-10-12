use super::*;

pub(super) struct HttpResponse {
    pub(super) status: u16,
    pub(super) content_type: Option<String>,
    pub(super) body: Vec<u8>,
}

pub(super) fn read_http_response(stream: &mut TcpStream) -> Result<HttpResponse, StoreError> {
    let mut encoded = Vec::with_capacity(4096);
    let header_end = loop {
        if let Some(position) = find_bytes(&encoded, b"\r\n\r\n") {
            break position + 4;
        }
        if encoded.len() >= MAX_HTTP_HEADER_BYTES {
            return Err(StoreError::State("OTLP HTTP response header is too large"));
        }
        read_more(stream, &mut encoded, MAX_HTTP_HEADER_BYTES)?;
    };
    let header = std::str::from_utf8(&encoded[..header_end - 4])
        .map_err(|_| StoreError::State("OTLP HTTP response header is not ASCII"))?;
    if !header.is_ascii() {
        return Err(StoreError::State("OTLP HTTP response header is not ASCII"));
    }
    let mut lines = header.split("\r\n");
    let status_line = lines
        .next()
        .ok_or(StoreError::State("OTLP HTTP response has no status"))?;
    let mut status_parts = status_line.split_ascii_whitespace();
    let version = status_parts.next().unwrap_or_default();
    let status = status_parts
        .next()
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or(StoreError::State("OTLP HTTP response status is invalid"))?;
    if !matches!(version, "HTTP/1.0" | "HTTP/1.1") || !(100..=599).contains(&status) {
        return Err(StoreError::State("OTLP HTTP response status is invalid"));
    }
    let mut content_length = None;
    let mut content_type = None;
    let mut chunked = false;
    for line in lines {
        let (name, value) = line
            .split_once(':')
            .ok_or(StoreError::State("OTLP HTTP response header is malformed"))?;
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim();
        match name.as_str() {
            "content-length" => {
                if content_length.is_some() {
                    return Err(StoreError::State(
                        "OTLP HTTP response repeats content-length",
                    ));
                }
                content_length = Some(value.parse::<usize>().map_err(|_| {
                    StoreError::State("OTLP HTTP response content-length is invalid")
                })?);
            }
            "content-type" => {
                if content_type.is_some() {
                    return Err(StoreError::State("OTLP HTTP response repeats content-type"));
                }
                content_type = Some(value.to_ascii_lowercase());
            }
            "content-encoding" if !value.eq_ignore_ascii_case("identity") => {
                return Err(StoreError::State(
                    "OTLP HTTP response encoding is unsupported",
                ));
            }
            "transfer-encoding" => {
                if !value.eq_ignore_ascii_case("chunked") || chunked {
                    return Err(StoreError::State(
                        "OTLP HTTP transfer encoding is unsupported",
                    ));
                }
                chunked = true;
            }
            _ => {}
        }
    }
    if chunked && content_length.is_some() {
        return Err(StoreError::State(
            "OTLP HTTP response has ambiguous body framing",
        ));
    }
    let initial_body = encoded.split_off(header_end);
    let body = if chunked {
        read_chunked_body(stream, initial_body)?
    } else if let Some(length) = content_length {
        if length > MAX_OTLP_HTTP_RESPONSE_BYTES {
            return Err(StoreError::State("OTLP HTTP response body is too large"));
        }
        read_exact_body(stream, initial_body, length)?
    } else {
        read_to_close(stream, initial_body)?
    };
    Ok(HttpResponse {
        status,
        content_type,
        body,
    })
}

fn read_more(
    stream: &mut TcpStream,
    destination: &mut Vec<u8>,
    maximum: usize,
) -> Result<(), StoreError> {
    let remaining = maximum.saturating_sub(destination.len());
    if remaining == 0 {
        return Err(StoreError::State("OTLP HTTP response exceeds its bound"));
    }
    let mut buffer = [0u8; 8192];
    let read_bound = remaining.min(buffer.len());
    let read = stream
        .read(&mut buffer[..read_bound])
        .map_err(io_error("read OTLP HTTP response"))?;
    if read == 0 {
        return Err(StoreError::State("OTLP HTTP response ended early"));
    }
    destination.extend_from_slice(&buffer[..read]);
    Ok(())
}

fn read_exact_body(
    stream: &mut TcpStream,
    mut body: Vec<u8>,
    length: usize,
) -> Result<Vec<u8>, StoreError> {
    if body.len() > length {
        return Err(StoreError::State(
            "OTLP HTTP response contains trailing bytes",
        ));
    }
    while body.len() < length {
        read_more(stream, &mut body, length)?;
    }
    Ok(body)
}

fn read_to_close(stream: &mut TcpStream, mut body: Vec<u8>) -> Result<Vec<u8>, StoreError> {
    if body.len() > MAX_OTLP_HTTP_RESPONSE_BYTES {
        return Err(StoreError::State("OTLP HTTP response body is too large"));
    }
    loop {
        let mut buffer = [0u8; 8192];
        let remaining = MAX_OTLP_HTTP_RESPONSE_BYTES.saturating_sub(body.len());
        if remaining == 0 {
            let mut extra = [0u8; 1];
            if stream
                .read(&mut extra)
                .map_err(io_error("read OTLP HTTP response"))?
                != 0
            {
                return Err(StoreError::State("OTLP HTTP response body is too large"));
            }
            return Ok(body);
        }
        let read_bound = remaining.min(buffer.len());
        let read = stream
            .read(&mut buffer[..read_bound])
            .map_err(io_error("read OTLP HTTP response"))?;
        if read == 0 {
            return Ok(body);
        }
        body.extend_from_slice(&buffer[..read]);
    }
}

fn read_chunked_body(stream: &mut TcpStream, mut encoded: Vec<u8>) -> Result<Vec<u8>, StoreError> {
    let mut cursor = 0usize;
    let mut decoded = Vec::new();
    loop {
        let line_end = loop {
            if let Some(relative) = find_bytes(&encoded[cursor..], b"\r\n") {
                break cursor + relative;
            }
            read_more(
                stream,
                &mut encoded,
                MAX_OTLP_HTTP_RESPONSE_BYTES + MAX_HTTP_HEADER_BYTES,
            )?;
        };
        let line = std::str::from_utf8(&encoded[cursor..line_end])
            .map_err(|_| StoreError::State("OTLP HTTP chunk size is invalid"))?;
        let size_text = line.split(';').next().unwrap_or_default();
        if size_text.is_empty() || size_text.len() > 16 {
            return Err(StoreError::State("OTLP HTTP chunk size is invalid"));
        }
        let size = usize::from_str_radix(size_text, 16)
            .map_err(|_| StoreError::State("OTLP HTTP chunk size is invalid"))?;
        cursor = line_end + 2;
        if size == 0 {
            while encoded.len() < cursor + 2 {
                read_more(
                    stream,
                    &mut encoded,
                    MAX_OTLP_HTTP_RESPONSE_BYTES + MAX_HTTP_HEADER_BYTES,
                )?;
            }
            if &encoded[cursor..cursor + 2] != b"\r\n" || encoded.len() != cursor + 2 {
                return Err(StoreError::State(
                    "OTLP HTTP chunk trailers are not accepted",
                ));
            }
            return Ok(decoded);
        }
        if decoded.len().checked_add(size).is_none()
            || decoded.len() + size > MAX_OTLP_HTTP_RESPONSE_BYTES
        {
            return Err(StoreError::State("OTLP HTTP response body is too large"));
        }
        let required = cursor
            .checked_add(size)
            .and_then(|value| value.checked_add(2))
            .ok_or(StoreError::State("OTLP HTTP chunk size overflows"))?;
        while encoded.len() < required {
            read_more(
                stream,
                &mut encoded,
                MAX_OTLP_HTTP_RESPONSE_BYTES + MAX_HTTP_HEADER_BYTES,
            )?;
        }
        if &encoded[cursor + size..required] != b"\r\n" {
            return Err(StoreError::State("OTLP HTTP chunk terminator is invalid"));
        }
        decoded.extend_from_slice(&encoded[cursor..cursor + size]);
        cursor = required;
    }
}

pub(super) fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}
